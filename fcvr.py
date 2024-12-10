# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pprint import pformat
from typing import List

import sys
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

import encoder
from encoder import *
from decoder import PixelDecoder, FeatureDecoder
import torch.nn.functional as F

class FCVR(nn.Module):
    def __init__(
            self, sparse_encoder: encoder.SparseEncoder, dense_decoder: PixelDecoder, momentum_encoder: MomentumEncoder, feature_decoder: FeatureDecoder,
            mask_ratio=0.6, temperature=0.07, densify_norm='bn', sbn=False,
    ):
        super().__init__()
        input_size, downsample_raito = sparse_encoder.input_size, sparse_encoder.downsample_raito
        self.downsample_raito = downsample_raito
        self.fmap_h, self.fmap_w = input_size // downsample_raito, input_size // downsample_raito
        self.mask_ratio = mask_ratio
        self.len_keep = round(self.fmap_h * self.fmap_w * (1 - mask_ratio))
        self.compute_contrastive_loss = nn.CrossEntropyLoss()
        self.temperature = temperature

        self.sparse_encoder = sparse_encoder
        self.dense_decoder = dense_decoder
        self.momentum_encoder = momentum_encoder
        self.feature_decoder = feature_decoder
        self.sbn = sbn
        self.hierarchy = len(sparse_encoder.enc_feat_map_chs)
        self.densify_norm_str = densify_norm.lower()
        # pixel decoder's densify layers
        self.pixel_densify_norms = nn.ModuleList()
        self.pixel_densify_projs = nn.ModuleList()
        self.pixel_mask_tokens = nn.ParameterList()
        # feature decoder's densify layers
        self.feature_mask_tokens = nn.ParameterList()
        self.feature_densify_norms = nn.ModuleList()
        self.feature_densify_projs = nn.ModuleList()
        # build the `densify` layers
        e_widths, d_width = self.sparse_encoder.enc_feat_map_chs, self.dense_decoder.width
        e_widths: List[int]
        for i in range(self.hierarchy): # from the smallest feat map to the largest; i=0: the last feat map; i=1: the second last feat map ...
            e_width = e_widths.pop()
            # create mask token
            p_p = nn.Parameter(torch.zeros(1, e_width, 1, 1))
            trunc_normal_(p_p, mean=0, std=.02, a=-.02, b=.02)
            self.pixel_mask_tokens.append(p_p)

            f_p = nn.Parameter(torch.zeros(1, e_width, 1, 1))
            trunc_normal_(f_p, mean=0, std=.02, a=-.02, b=.02)
            self.feature_mask_tokens.append(f_p)

            
            # create densify norm
            if self.densify_norm_str == 'bn':
                p_densify_norm = (encoder.SparseSyncBatchNorm2d if self.sbn else encoder.SparseBatchNorm2d)(e_width)
                f_densify_norm = (encoder.SparseSyncBatchNorm2d if self.sbn else encoder.SparseBatchNorm2d)(e_width)
            elif self.densify_norm_str == 'ln':
                p_densify_norm = encoder.SparseConvNeXtLayerNorm(e_width, data_format='channels_first', sparse=True)
                f_densify_norm = encoder.SparseConvNeXtLayerNorm(e_width, data_format='channels_first', sparse=True)
            else:
                p_densify_norm = nn.Identity()
                f_densify_norm = nn.Identity()
            self.pixel_densify_norms.append(p_densify_norm)
            self.feature_densify_norms.append(f_densify_norm)
            
            # create densify proj
            if i == 0 and e_width == d_width:
                p_densify_proj = nn.Identity()
                f_densify_proj = nn.Identity()
                print(f'[FCVR.__init__, densify {i+1}/{self.hierarchy}]: use nn.Identity() as densify_proj')
            else:
                kernel_size = 1 if i <= 0 else 3
                p_densify_proj = nn.Conv2d(e_width, d_width, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True)
                f_densify_proj = nn.Conv2d(e_width, d_width, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True)
                print(f'[FCVR.__init__, densify {i+1}/{self.hierarchy}]: densify_proj(ksz={kernel_size}, #para={sum(x.numel() for x in p_densify_proj.parameters()) / 1e6:.2f}M)')
            self.pixel_densify_projs.append(p_densify_proj)
            self.feature_densify_projs.append(f_densify_proj)
            
            # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
            d_width //= 2
        
        print(f'[FCVR.__init__] dims of mask_tokens={tuple(p.numel() for p in self.pixel_mask_tokens)}')
        
    
    def mask(self, B: int, device, generator=None):
        h, w = self.fmap_h, self.fmap_w
        idx = torch.rand(B, h * w, generator=generator).argsort(dim=1)
        idx = idx[:, :self.len_keep].to(device)  # (B, len_keep)
        return torch.zeros(B, h * w, dtype=torch.bool, device=device).scatter_(dim=1, index=idx, value=True).view(B, 1, h, w)
    
    def cosine_similarity(self, x: torch.Tensor, y: torch.Tensor):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return torch.matmul(x, y.T)
        
    def forward(self, inp_bchw_b1: torch.Tensor, inp_bchw_b2: torch.Tensor, active_b1ff=None, vis=False): 
        # step1. Mask
        if active_b1ff is None:     # rand mask
            active_b1ff: torch.BoolTensor = self.mask(inp_bchw_b1.shape[0], inp_bchw_b1.device)  # (B, 1, f, f)
        encoder._cur_active = active_b1ff    # (B, 1, f, f)
        active_b1hw = active_b1ff.repeat_interleave(self.downsample_raito, 2).repeat_interleave(self.downsample_raito, 3)  # (B, 1, H, W)
        masked_bchw = inp_bchw_b1 * active_b1hw

        # step2. Encode: get hierarchical encoded sparse features (a list containing 4 feature maps at 4 scales)
        sp_fea_bcffs: List[torch.Tensor] = self.sparse_encoder(masked_bchw)
        sp_fea_bcffs.reverse()  # after reversion: from the smallest feature map to the largest
        
        # step3. Densify for pixel decoder
        cur_active = active_b1ff     # (B, 1, f, f)
        to_pix_dec = []
        for i, bcff in enumerate(sp_fea_bcffs):  # from the smallest feature map to the largest
            if bcff is not None:
                bcff = self.pixel_densify_norms[i](bcff)
                mask_tokens = self.pixel_mask_tokens[i].expand_as(bcff)
                bcff = torch.where(cur_active.expand_as(bcff), bcff, mask_tokens)   # fill in empty (non-active) positions with [mask] tokens
                bcff: torch.Tensor = self.pixel_densify_projs[i](bcff)
            to_pix_dec.append(bcff)
            cur_active = cur_active.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)  # dilate the mask map, from (B, 1, f, f) to (B, 1, H, W)
        
        # step4. Decode and reconstruct
        rec_bchw = self.dense_decoder(to_pix_dec)
        inp, rec = self.patchify(inp_bchw_b1), self.patchify(rec_bchw)   # inp and rec: (B, L = f*f, N = C*downsample_raito**2)
        mean = inp.mean(dim=-1, keepdim=True)
        var = (inp.var(dim=-1, keepdim=True) + 1e-6) ** .5
        inp = (inp - mean) / var
        l2_loss = ((rec - inp) ** 2).mean(dim=2, keepdim=False)    # (B, L, C) ==mean==> (B, L)
        non_active = active_b1ff.logical_not().int().view(active_b1ff.shape[0], -1)  # (B, 1, f, f) => (B, L)
        recon_loss = l2_loss.mul_(non_active).sum() / (non_active.sum() + 1e-8)  # loss only on masked (non-active) patches
        

        # step5. encode: Momentum encoder
        with torch.no_grad():
            mo_fea_bc11s: List[torch.Tensor] = self.momentum_encoder(inp_bchw_b2) # after gap feature (B, C, 1, 1)

        # step6. Densify for feature decoder
        cur_active = active_b1ff     # (B, 1, f, f)
        to_fea_dec = []
        for i, bcff in enumerate(sp_fea_bcffs):  # from the smallest feature map to the largest
            if bcff is not None:
                bcff = self.feature_densify_norms[i](bcff)
                mask_tokens = self.feature_mask_tokens[i].expand_as(bcff)
                bcff = torch.where(cur_active.expand_as(bcff), bcff, mask_tokens)   # fill in empty (non-active) positions with [mask] tokens
                bcff: torch.Tensor = self.feature_densify_projs[i](bcff)
            to_fea_dec.append(bcff)
            cur_active = cur_active.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)  # dilate the mask map, from (B, 1, f, f) to (B, 1, H, W)

        # step7. Decode for features and contrastive loss
        con_bc11s: List[torch.Tensor] = self.feature_decoder(to_fea_dec) # after gap feature (B, C, 1, 1)
        contrastive_loss = torch.tensor(0.).cuda()
        for i in range(self.hierarchy):
            mo_fea_bc = mo_fea_bc11s[i].view(mo_fea_bc11s[i].shape[:2])  # (B, C, 1, 1) => (B, C)
            con_bc = con_bc11s[i].view(con_bc11s[i].shape[:2])  # (B, C, 1, 1) => (B, C)
            similarity = self.cosine_similarity(con_bc, mo_fea_bc)  # (B, B)
            label = torch.arange(similarity.shape[0]).cuda()
            contrastive_loss += self.compute_contrastive_loss(similarity / self.temperature, label) # (B, B) => scalar


        if vis:
            masked_bchw = inp_bchw_b1 * active_b1hw
            rec_bchw = self.unpatchify(rec * var + mean)
            rec_or_inp = torch.where(active_b1hw, inp_bchw_b1, rec_bchw)
            return inp_bchw_b1, masked_bchw, rec_or_inp
        else:
            masked_bchw = inp_bchw_b1 * active_b1hw
            rec_bchw = self.unpatchify(rec * var + mean)
            rec_or_inp = torch.where(active_b1hw, inp_bchw_b1, rec_bchw)
            import cv2
            cv2.imencode('.jpg', inp_bchw_b1[0].permute(1,2,0).detach().cpu().numpy() * 255)[1].tofile(r'D:\论文\我的论文\MAE\FCVR\visual_img\0.bmp')
            cv2.imencode('.jpg', masked_bchw[0].permute(1,2,0).detach().cpu().numpy() * 255)[1].tofile(r'D:\论文\我的论文\MAE\FCVR\visual_img\1.bmp')
            cv2.imencode('.jpg', rec_bchw[0].permute(1,2,0).detach().cpu().numpy() * 255)[1].tofile(r'D:\论文\我的论文\MAE\FCVR\visual_img\2.bmp')
            cv2.imencode('.jpg', rec_or_inp[0].permute(1,2,0).detach().cpu().numpy() * 255)[1].tofile(r'D:\论文\我的论文\MAE\FCVR\visual_img\3.bmp')
            return recon_loss + 0.25 * contrastive_loss, inp_bchw_b1, masked_bchw, rec_or_inp
    
    def patchify(self, bchw):
        p = self.downsample_raito
        h, w = self.fmap_h, self.fmap_w
        B, C = bchw.shape[:2]
        bchw = bchw.reshape(shape=(B, C, h, p, w, p))
        bchw = torch.einsum('bchpwq->bhwpqc', bchw)
        bln = bchw.reshape(shape=(B, h * w, C * p ** 2))  # (B, f*f, 3*downsample_raito**2)
        return bln
    
    def unpatchify(self, bln):
        p = self.downsample_raito
        h, w = self.fmap_h, self.fmap_w
        B, C = bln.shape[0], bln.shape[-1] // p ** 2
        bln = bln.reshape(shape=(B, h, w, p, p, C))
        bln = torch.einsum('bhwpqc->bchpwq', bln)
        bchw = bln.reshape(shape=(B, C, h * p, w * p))
        return bchw
    
    def __repr__(self):
        return (
            f'\n'
            f'[FCVR.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[FCVR.structure]: {super(FCVR, self).__repr__().replace(FCVR.__name__, "")}'
        )
    
    def get_config(self):
        return {
            # self
            'mask_ratio': self.mask_ratio,
            'densify_norm_str': self.densify_norm_str,
            'sbn': self.sbn, 'hierarchy': self.hierarchy,
            
            # enc
            'sparse_encoder.input_size': self.sparse_encoder.input_size,
            # dec
            'dense_decoder.width': self.dense_decoder.width,
        }
    
    def state_dict(self, destination=None, prefix='', keep_vars=False, with_config=False):
        state = super(FCVR, self).state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if with_config:
            state['config'] = self.get_config()
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        config: dict = state_dict.pop('config', None)
        incompatible_keys = super(FCVR, self).load_state_dict(state_dict, strict=strict)
        if config is not None:
            for k, v in self.get_config().items():
                ckpt_v = config.get(k, None)
                if ckpt_v != v:
                    err = f'[SparseMIM.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={ckpt_v})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err, file=sys.stderr)
        return incompatible_keys
