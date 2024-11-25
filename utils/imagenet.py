# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Callable, Optional, Tuple
from utils.augment import non_spatial_displacement_method, spatial_displacement_method
import cv2
import numpy as np
import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
try:
    from torchvision.transforms import InterpolationMode
    interpolation = InterpolationMode.BICUBIC
except:
    import PIL
    interpolation = PIL.Image.BICUBIC


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f: img: PImage.Image = PImage.open(f).convert('RGB')
    return img

class YOLODataset(VisionDataset):
    def __init__(self, path_list, 
                 non_spatial_dis, spatial_dis, formats) -> None:
        super().__init__()
        self.path_list = path_list if isinstance(path_list, list) else [path_list]
        self.non_spatial_dis = non_spatial_dis
        self.spatial_dis = spatial_dis
        self.format = formats
        self.samples = self._mksample()

    def _mksample(self):
        samples = []
        for path in self.path_list:
            for f in os.listdir(path):
                if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.JPEG') or f.endswith('.JPG') or f.endswith('.bmp'):
                    img_path = os.path.join(path, f)
                    samples.append(img_path)
                else:
                    continue
        return samples

    def __getitem__(self, index: int):
        img_path = self.samples[index]
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = self.spatial_dis(image)
        im1, im2 = self.microtranslation(image)
        im2 = self.non_spatial_dis(image)
        return self.format(im1), self.format(im2)
        
        # # 大尺度空间位移对对比生成模型的影响。
        # img_path = self.samples[index]
        # image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        # im1 = self.spatial_dis(image)
        # im2 = self.spatial_dis(image)
        # # cv2.imencode('.jpg', im1)[1].tofile(r'D:\论文\我的论文\MAE\FCVR_code\visual_img\spatialdis1.jpg')
        # # cv2.imencode('.jpg', im2)[1].tofile(r'D:\论文\我的论文\MAE\FCVR_code\visual_img\spatialdis2.jpg')
        # im2 = self.non_spatial_dis(im2)
        # # cv2.imencode('.jpg', im2)[1].tofile(r'D:\论文\我的论文\MAE\FCVR_code\visual_img\nonspatialdis2.jpg')
        # return self.format(im1), self.format(im2)

    def __len__(self) -> int:
        return len(self.samples)
    
    def microtranslation(self, image):
        scale = (0.5, 1.0)
        ratio = (0.9, 1.11)
        min_shift, max_shift = -10, 10
        h, w = image.shape[:2]
        target_area = np.random.uniform(*scale) * h * w
        aspect_ratio = np.random.uniform(*ratio)
        crop_h = int(round(np.sqrt(target_area * aspect_ratio)))
        crop_w = int(round(np.sqrt(target_area / aspect_ratio)))

        crop_h = crop_h if crop_h < h else h - 10
        crop_w = crop_w if crop_w < w else w - 10

        center_x = np.random.randint(int(crop_h / 2), int(h - crop_h / 2)) # compute baseline image's center
        center_y = np.random.randint(int(crop_w / 2), int(w - crop_w / 2))

        min_shift_x = int(crop_h / 2 - center_x) if abs(min_shift) > abs(int(crop_h / 2 - center_x)) else min_shift
        max_shift_x = int(h - crop_h / 2 - center_x) if abs(max_shift) > abs(int(h - crop_h / 2 - center_x)) else max_shift

        min_shift_y = int(crop_w / 2 - center_y) if abs(min_shift) > abs(int(crop_w / 2 - center_y)) else min_shift
        max_shift_y = int(w - crop_w / 2 - center_y) if abs(max_shift) > abs(int(w - crop_w / 2 - center_y)) else max_shift

        delta_x1, delta_x2 = (np.random.randint(min_shift_x, max_shift_x) for _ in range(2)) # compute the displacement
        delta_y1, delta_y2 = (np.random.randint(min_shift_y, max_shift_y) for _ in range(2))
        
        crop1 = image[center_x - crop_h // 2 + delta_x1: center_x + crop_h // 2 + delta_x1, center_y - crop_w // 2 + delta_y1: center_y + crop_w // 2 + delta_y1]
        crop2 = image[center_x - crop_h // 2 + delta_x2: center_x + crop_h // 2 + delta_x2, center_y - crop_w // 2 + delta_y2: center_y + crop_w // 2 + delta_y2]
        
        return crop1, crop2 
class ImageNetDataset(DatasetFolder):
    def __init__(
            self,
            imagenet_folder: str,
            train: bool,
            transform: Callable,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        imagenet_folder = os.path.join(imagenet_folder, 'train' if train else 'val')
        super(ImageNetDataset, self).__init__(
            imagenet_folder,
            loader=pil_loader,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=None, is_valid_file=is_valid_file
        )
        
        self.samples = tuple(img for (img, label) in self.samples)
        self.targets = None # this is self-supervised learning so we don't need labels
    
    def __getitem__(self, index: int) -> Any:
        img_file_path = self.samples[index]
        return self.transform(self.loader(img_file_path))


def build_dataset_to_pretrain(dataset_path, input_size, hyp) -> Dataset:
    """
    You may need to modify this function to return your own dataset.
    Define a new class, a subclass of `Dataset`, to replace our ImageNetDataset.
    Use dataset_path to build your image file path list.
    Use input_size to create the transformation function for your images, can refer to the `trans_train` blow. 
    
    :param dataset_path: the folder of dataset
    :param input_size: the input size (image resolution)
    :return: the dataset used for pretraining
    """
    with open(hyp, 'r') as f:
        import yaml
        hyp = yaml.safe_load(f)
    formatation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.273, 0.273, 0.273), std=(0.256, 0.256, 0.256)),
    ])
    non_spatial_dis = non_spatial_displacement_method(hyp)
    spatial_dis = spatial_displacement_method(hyp)
    dataset_train = YOLODataset(path_list=dataset_path, non_spatial_dis=non_spatial_dis, spatial_dis=spatial_dis, formats=formatation)
    print_transform(non_spatial_dis, '[pre-train]')
    print_transform(spatial_dis, '[pre-train]')
    return dataset_train


def print_transform(transform, s):
    print(f'Transform {s} = ')
    for t in transform.transforms:
        print(t)
    print('---------------------------\n')

