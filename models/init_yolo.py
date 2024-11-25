# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# import sys


from models.v8.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    WorldDetect,
    v10Detect,
)
# from models.v5.modules import (AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
#                                     Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d,
#                                     Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, RepC3, RepConv,
#                                     )
from models.v8 import LOGGER, colorstr, yaml_model_load
import torch
import torch.nn as nn
from typing import List
from timm.models.registry import register_model
import contextlib
import math 


class YOLO(nn.Module):
    """
    This is a template for your custom ConvNet.
    It is required to implement the following three functions: `get_downsample_ratio`, `get_feature_map_channels`, `forward`.
    You can refer to the implementations in `pretrain\models\resnet.py` for an example.
    """
    def __init__(self, ym_path, num_classes, global_pool) -> None:
        super().__init__()
        self.model_dict = yaml_model_load(ym_path)
        self.model = self.parse_model(self.model_dict, ch=3)
        self.initialize_weights()

    ## yolov5
    # def parse_model(self, d, ch):
    #     """Parses a YOLOv5 model from a dict `d`, configuring layers based on input channels `ch` and model architecture."""
    #     LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    #     anchors, nc, gd, gw, act, ch_mul = (
    #         d["anchors"],
    #         d["nc"],
    #         d["depth_multiple"],
    #         d["width_multiple"],
    #         d.get("activation"),
    #         d.get("channel_multiple"),
    #     )
    #     if act:
    #         Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
    #         LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    #     if not ch_mul:
    #         ch_mul = 8
    #     na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    #     no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    #     layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    #     for i, (f, n, m, args) in enumerate(d["backbone"]):  # from, number, module, args
    #         m = eval(m) if isinstance(m, str) else m  # eval strings
    #         for j, a in enumerate(args):
    #             with contextlib.suppress(NameError):
    #                 args[j] = eval(a) if isinstance(a, str) else a  # eval strings

    #         n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
    #         if m in {
    #             Conv,
    #             GhostConv,
    #             Bottleneck,
    #             GhostBottleneck,
    #             SPP,
    #             SPPF,
    #             DWConv,
    #             Focus,
    #             BottleneckCSP,
    #             C3,
    #             C3TR,
    #             C3Ghost,
    #             nn.ConvTranspose2d,
    #             DWConvTranspose2d,
    #             C3x,
    #         }:
    #             c1, c2 = ch[f], args[0]
    #             if c2 != no:  # if not output
    #                 c2 = self.make_divisible(c2 * gw, ch_mul)

    #             args = [c1, c2, *args[1:]]
    #             if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
    #                 args.insert(2, n)  # number of repeats
    #                 n = 1
    #         elif m is nn.BatchNorm2d:
    #             args = [ch[f]]
    #         elif m is Concat:
    #             c2 = sum(ch[x] for x in f)
    #         else:
    #             c2 = ch[f]

    #         m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
    #         t = str(m)[8:-2].replace("__main__.", "")  # module type
    #         np = sum(x.numel() for x in m_.parameters())  # number params
    #         m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
    #         LOGGER.info(f"{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}")  # print
    #         save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
    #         layers.append(m_)
    #         if i == 0:
    #             ch = []
    #         ch.append(c2)
    #     return nn.Sequential(*layers)

    def parse_model(self, d, ch=3, verbose=True):  # model_dict, input_channels(3)
        """Parse a YOLO model.yaml dictionary into a PyTorch model."""
        import ast

        # Args
        legacy = True  # backward compatibility for v3/v5/v8/v9 models
        max_channels = float("inf")
        nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
        depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
        if scales:
            scale = d.get("scale")
            if not scale:
                scale = tuple(scales.keys())[0]
                LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
            depth, width, max_channels = scales[scale]

        if act:
            Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
            if verbose:
                LOGGER.info(f"{colorstr('activation:')} {act}")  # print

        if verbose:
            LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
        ch = [ch]
        layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
        for i, (f, n, m, args) in enumerate(d["backbone"]):  # from, number, module, args
            m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
            for j, a in enumerate(args):
                if isinstance(a, str):
                    try:
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
                    except ValueError:
                        pass
            n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
            if m in {
                Classify,
                Conv,
                ConvTranspose,
                GhostConv,
                Bottleneck,
                GhostBottleneck,
                SPP,
                SPPF,
                C2fPSA,
                C2PSA,
                DWConv,
                Focus,
                BottleneckCSP,
                C1,
                C2,
                C2f,
                C3k2,
                RepNCSPELAN4,
                ELAN1,
                ADown,
                AConv,
                SPPELAN,
                C2fAttn,
                C3,
                C3TR,
                C3Ghost,
                nn.ConvTranspose2d,
                DWConvTranspose2d,
                C3x,
                RepC3,
                PSA,
                SCDown,
                C2fCIB,
            }:
                c1, c2 = ch[f], args[0]
                if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                    c2 = self.make_divisible(min(c2, max_channels) * width, 8)
                if m is C2fAttn:
                    args[1] = self.make_divisible(min(args[1], max_channels // 2) * width, 8)  # embed channels
                    args[2] = int(
                        max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
                    )  # num heads

                args = [c1, c2, *args[1:]]
                if m in {
                    BottleneckCSP,
                    C1,
                    C2,
                    C2f,
                    C3k2,
                    C2fAttn,
                    C3,
                    C3TR,
                    C3Ghost,
                    C3x,
                    RepC3,
                    C2fPSA,
                    C2fCIB,
                    C2PSA,
                }:
                    args.insert(2, n)  # number of repeats
                    n = 1
                if m is C3k2:  # for M/L/X sizes
                    legacy = False
                    if scale in "mlx":
                        args[3] = True
            elif m is AIFI:
                args = [ch[f], *args]
            elif m in {HGStem, HGBlock}:
                c1, cm, c2 = ch[f], args[0], args[1]
                args = [c1, cm, c2, *args[2:]]
                if m is HGBlock:
                    args.insert(4, n)  # number of repeats
                    n = 1
            elif m is ResNetLayer:
                c2 = args[1] if args[3] else args[1] * 4
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum(ch[x] for x in f)
            elif m in {Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect}:
                args.append([ch[x] for x in f])
                if m is Segment:
                    args[2] = self.make_divisible(min(args[2], max_channels) * width, 8)
                if m in {Detect, Segment, Pose, OBB}:
                    m.legacy = legacy
            elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
                args.insert(1, [ch[x] for x in f])
            elif m is CBLinear:
                c2 = args[0]
                c1 = ch[f]
                args = [c1, c2, *args[1:]]
            elif m is CBFuse:
                c2 = ch[f[-1]]
            else:
                c2 = ch[f]

            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
            t = str(m)[8:-2].replace("__main__.", "")  # module type
            m_.np = sum(x.numel() for x in m_.parameters())  # number params
            m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
            if verbose:
                LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # print
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
        return nn.Sequential(*layers)

    def initialize_weights(self):
        """Initialize model weights to random values."""
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True
    
    def make_divisible(self, x, divisor):
        """
        Returns the nearest number that is divisible by the given divisor.

        Args:
            x (int): The number to make divisible.
            divisor (int | torch.Tensor): The divisor.

        Returns:
            (int): The nearest number divisible by the divisor.
        """
        if isinstance(divisor, torch.Tensor):
            divisor = int(divisor.max())  # to int
        return math.ceil(x / divisor) * divisor

    # def make_divisible(self, x, divisor):             # yolov5
    #     """Returns nearest x divisible by divisor."""
    #     if isinstance(divisor, torch.Tensor):
    #         divisor = int(divisor.max())  # to int
    #     return math.ceil(x / divisor) * divisor

    def get_downsample_ratio(self) -> int:
        # max downsampling rate
        return 32
    
    def get_feature_map_channels(self) -> List[int]:
        base_channels = [256, 512, 1024]
        scales = {
            'n' : [0.25, 1024],
            's' : [0.50, 1024],
            'm' : [0.75, 768],
            'l' : [1.00, 512],
            'x' : [1.25, 512],
            }
        
        scale = scales[self.model_dict['scale']][0]
        max_channel = scales[self.model_dict['scale']][1]
        return [int(min(max_channel, c) * scale) for c in base_channels]
    
    # def get_feature_map_channels(self) -> List[int]:
    #     return [192, 384, 768]
    
    def forward(self, x: torch.Tensor, hierarchical=False):
        #TODO: if or not FPN+PAN
        y = []
        for ix, m in enumerate(self.model):
            x = m(x)
            if ix in [4, 6, 9]:
                y.append(x)
        return y
    


@register_model
def yolo(pretrained=False, **kwargs):
    return YOLO(**kwargs)


@torch.no_grad()
def convnet_test():
    from timm.models import create_model
    cnn = create_model('yolov5l', ym_path=r'models/yolov5/model/yolov5l-seg.yaml')
    print('get_downsample_ratio:', cnn.get_downsample_ratio())
    print('get_feature_map_channels:', cnn.get_feature_map_channels())
    
    downsample_ratio = cnn.get_downsample_ratio()
    feature_map_channels = cnn.get_feature_map_channels()
    
    # check the forward function
    B, C, H, W = 4, 3, 640, 640
    inp = torch.rand(B, C, H, W)
    feats = cnn(inp, hierarchical=True)
    assert isinstance(feats, list)
    assert len(feats) == len(feature_map_channels)
    print([tuple(t.shape) for t in feats])
    
    # check the downsample ratio
    feats = cnn(inp, hierarchical=True)
    assert feats[-1].shape[-2] == H // downsample_ratio
    assert feats[-1].shape[-1] == W // downsample_ratio
    
    # check the channel number
    for feat, ch in zip(feats, feature_map_channels):
        assert feat.ndim == 4
        assert feat.shape[1] == ch


if __name__ == '__main__':
    convnet_test()
