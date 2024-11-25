# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import random
from copy import deepcopy

import cv2
import numpy as np
import torch
import torchvision.transforms as T

# from ultralytics.utils import LOGGER, colorstr
# from ultralytics.utils.checks import check_version
# from ultralytics.utils.instance import Instances

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# TODO: we might need a BaseTransform to make all these augments be compatible with both classification and semantic
class BaseTransform:
    """
    Base class for image transformations.

    This is a generic transformation class that can be extended for specific image processing needs.
    The class is designed to be compatible with both classification and semantic segmentation tasks.

    Methods:
        __init__: Initializes the BaseTransform object.
        apply_image: Applies image transformation to labels.
        apply_instances: Applies transformations to object instances in labels.
        apply_semantic: Applies semantic segmentation to an image.
        __call__: Applies all label transformations to an image, instances, and semantic masks.
    """

    def __init__(self) -> None:
        """Initializes the BaseTransform object."""
        pass

    def apply_image(self, labels):
        """Applies image transformations to labels."""
        pass

    def apply_instances(self, labels):
        """Applies transformations to object instances in labels."""
        pass

    def apply_semantic(self, labels):
        """Applies semantic segmentation to an image."""
        pass

    def __call__(self, labels):
        """Applies all label transformations to an image, instances, and semantic masks."""
        self.apply_image(labels)
        self.apply_instances(labels)
        self.apply_semantic(labels)


class Compose:
    """Class for composing multiple image transformations."""

    def __init__(self, transforms):
        """Initializes the Compose object with a list of transforms."""
        self.transforms = transforms

    def __call__(self, data):
        """Applies a series of transformations to input data."""
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform):
        """Appends a new transform to the existing list of transforms."""
        self.transforms.append(transform)

    def tolist(self):
        """Converts the list of transforms to a standard Python list."""
        return self.transforms

    def __repr__(self):
        """Returns a string representation of the object."""
        return f"{self.__class__.__name__}({', '.join([f'{t}' for t in self.transforms])})"


class BaseMixTransform:
    """
    Class for base mix (MixUp/Mosaic) transformations.

    This implementation is from mmyolo.
    """

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        """Initializes the BaseMixTransform object with dataset, pre_transform, and probability."""
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p

    def __call__(self, labels):
        """Applies pre-processing transforms and mixup/mosaic transforms to labels data."""
        if random.uniform(0, 1) > self.p:
            return labels

        # Get index of one or three other images
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        # Get images information will be used for Mosaic or MixUp
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels['mix_labels'] = mix_labels

        # Mosaic or MixUp
        labels = self._mix_transform(labels)
        labels.pop('mix_labels', None)
        return labels

    def _mix_transform(self, labels):
        """Applies MixUp or Mosaic augmentation to the label dictionary."""
        raise NotImplementedError

    def get_indexes(self):
        """Gets a list of shuffled indexes for mosaic augmentation."""
        raise NotImplementedError


class RandomPerspective:
    """
    Implements random perspective and affine transformations on images and corresponding bounding boxes, segments, and
    keypoints. These transformations include rotation, translation, scaling, and shearing. The class also offers the
    option to apply these transformations conditionally with a specified probability.

    Attributes:
        degrees (float): Degree range for random rotations.
        translate (float): Fraction of total width and height for random translation.
        scale (float): Scaling factor interval, e.g., a scale factor of 0.1 allows a resize between 90%-110%.
        shear (float): Shear intensity (angle in degrees).
        perspective (float): Perspective distortion factor.
        border (tuple): Tuple specifying mosaic border.
        pre_transform (callable): A function/transform to apply to the image before starting the random transformation.

    Methods:
        affine_transform(img, border): Applies a series of affine transformations to the image.
        apply_bboxes(bboxes, M): Transforms bounding boxes using the calculated affine matrix.
        apply_segments(segments, M): Transforms segments and generates new bounding boxes.
        apply_keypoints(keypoints, M): Transforms keypoints.
        __call__(labels): Main method to apply transformations to both images and their corresponding annotations.
        box_candidates(box1, box2): Filters out bounding boxes that don't meet certain criteria post-transformation.
    """

    def __init__(self,
                 rot90=False,
                 degrees=0.0,
                 translate=0.1,
                 scale=0.5,
                 shear=0.0,
                 perspective=0.0,
                 border=(0, 0),
                 pre_transform=None):
        """Initializes RandomPerspective object with transformation parameters."""
        self.rot90 = rot90 # add by fmr
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border  # mosaic border
        self.pre_transform = pre_transform

    def __call__(self, img):
        """
        Applies a sequence of affine transformations centered around the image center.

        Args:
            img (ndarray): Input image.
            border (tuple): Border dimensions.

        Returns:
            img (ndarray): Transformed image.
            M (ndarray): Transformation matrix.
            s (float): Scale factor.
        """

        # Center
        C = np.eye(3, dtype=np.float32)

        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations

        # add by fmr for rotation of 90.
        if  self.rot90: 
            a += random.choice([-180, -90, 0, 90])

        s = random.uniform(0.7, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * img.shape[1]  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * img.shape[0]  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        # Affine image
        if (self.border[0] != 0) or (self.border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=(img.shape[1], img.shape[0]), borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=(img.shape[1], img.shape[0]), borderValue=(114, 114, 114))
        return img


class RandomHSV:
    """
    This class is responsible for performing random adjustments to the Hue, Saturation, and Value (HSV) channels of an
    image.

    The adjustments are random but within limits set by hgain, sgain, and vgain.
    """

    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5) -> None:
        """
        Initialize RandomHSV class with gains for each HSV channel.

        Args:
            hgain (float, optional): Maximum variation for hue. Default is 0.5.
            sgain (float, optional): Maximum variation for saturation. Default is 0.5.
            vgain (float, optional): Maximum variation for value. Default is 0.5.
        """
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, img):
        """
        Applies random HSV augmentation to an image within the predefined limits.

        The modified image replaces the original image in the input 'labels' dict.
        """
        if self.hgain or self.sgain or self.vgain:
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        return img


class RandomFlip:
    """
    Applies a random horizontal or vertical flip to an image with a given probability.

    Also updates any instances (bounding boxes, keypoints, etc.) accordingly.
    """

    def __init__(self, p=0.5, direction='horizontal', flip_idx=None) -> None:
        """
        Initializes the RandomFlip class with probability and direction.

        Args:
            p (float, optional): The probability of applying the flip. Must be between 0 and 1. Default is 0.5.
            direction (str, optional): The direction to apply the flip. Must be 'horizontal' or 'vertical'.
                Default is 'horizontal'.
            flip_idx (array-like, optional): Index mapping for flipping keypoints, if any.
        """
        assert direction in ['horizontal', 'vertical'], f'Support direction `horizontal` or `vertical`, got {direction}'
        assert 0 <= p <= 1.0

        self.p = p
        self.direction = direction
        self.flip_idx = flip_idx

    def __call__(self, img):
        """
        Applies random flip to an image and updates any instances like bounding boxes or keypoints accordingly.

        Args:
            labels (dict): A dictionary containing the keys 'img' and 'instances'. 'img' is the image to be flipped.
                           'instances' is an object containing bounding boxes and optionally keypoints.

        Returns:
            (dict): The same dict with the flipped image and updated instances under the 'img' and 'instances' keys.
        """
        h, w = img.shape[:2]
        # Flip up-down
        if self.direction == 'vertical' and random.random() < self.p:
            img = np.flipud(img)
        if self.direction == 'horizontal' and random.random() < self.p:
            img = np.fliplr(img)
            # For keypoints
        img = np.ascontiguousarray(img)
        return img



class Albumentations:
    """
    Albumentations transformations.

    Optional, uninstall package to disable. Applies Blur, Median Blur, convert to grayscale, Contrast Limited Adaptive
    Histogram Equalization, random change of brightness and contrast, RandomGamma and lowering of image quality by
    compression.
    """

    def __init__(self, p=1.0):
        """Initialize the transform object for YOLO bbox formatted params."""
        self.p = p
        self.transform = None
        import albumentations as A

        T = [
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            # A.RandomResizedCrop(height=640, width=640, scale=(0.5, 1.0), ratio=(0.9, 1.11), p=0.5),
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(p=0.0),
            A.RandomGamma(p=0.0),
            A.ImageCompression(quality_lower=75, p=0.0)]  # transforms
        self.transform = A.Compose(T)


    def __call__(self, im):
        if self.transform and random.random() < self.p:
            im = self.transform(image=im)['image']  # transformed
        return im


def non_spatial_displacement_method(hyp):
    """Convert images to a size suitable for YOLOv8 training."""

    return Compose([
        Albumentations(p=hyp['albumentations']),
        RandomHSV(hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])])
        # RandomPerspective(
        #     rot90=hyp['rot90'],
        #     degrees=hyp['degrees'],
        #     translate=hyp['translate'],
        #     scale=hyp['scale'],
        #     shear=hyp['shear'],
        #     perspective=hyp['perspective'],
        #     pre_transform=None,
        # ),
        # RandomFlip(direction='vertical', p=hyp['flipud']),
        # RandomFlip(direction='horizontal', p=hyp['fliplr'], flip_idx=[])])  # transforms

def spatial_displacement_method(hyp):
    """Convert images to a size suitable for YOLOv8 training."""

    return Compose([
        # Albumentations(p=hyp['albumentations']),
        # RandomHSV(hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])])
        RandomPerspective(
            rot90=hyp['rot90'],
            degrees=hyp['degrees'],
            translate=hyp['translate'],
            scale=hyp['scale'],
            shear=hyp['shear'],
            perspective=hyp['perspective'],
            pre_transform=None,
        ),
        RandomFlip(direction='vertical', p=hyp['flipud']),
        RandomFlip(direction='horizontal', p=hyp['fliplr'], flip_idx=[])])  # transforms