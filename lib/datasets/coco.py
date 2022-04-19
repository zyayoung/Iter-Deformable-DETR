# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
from config import config
import os.path as osp
import json
import torch
import torch.utils.data
import torchvision
from config import config
from pycocotools import mask as coco_mask
from datasets.cocodet import CocoDetection
import datasets.transforms as T
import pdb


class CocoDetection(CocoDetection):
    def __init__(self, img_folder, ann_file, phase, transforms, return_masks):

        super(CocoDetection, self).__init__(img_folder, ann_file, phase)
        self._transforms = transforms
        assert phase in ['train', 'val']
        self.phase = phase
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def _filter_ignores(self, target):

        # annotations = target['annotations']
        # cates = np.array([rb['category_id'] for rb in annotations])
        target = list(filter(lambda rb: rb['category_id'] > -1, target))
        # target['annotations'] = annotations
        return target

    def _minus_target_label(self, target, value):

        results = []
        for t in target:
            t['category_id'] -= value
            results.append(t)
        return results

    def __getitem__(self, idx):
        
        img, target, imgname = super(CocoDetection, self).__getitem__(idx)

        target = self._minus_target_label(target, 1)
        total = len(target)
        image_id = self.ids[idx]
        
        target = {'image_id': image_id, 'annotations': target}

        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target, imgname


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.BoolTensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        iscrowd |= classes != 0

        target = {}
        target["boxes"] = boxes[keep]
        target["labels"] = classes[keep]
        if self.return_masks:
            target["masks"] = masks[keep]
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints[keep]

        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):
    
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def construct_dataset(ann_file, image_set, args):

    imgDir = '/home/zhenganlin/june/CrowdHuman/images'
    dataset = CocoDetection(imgDir, ann_file, image_set,
        transforms = make_coco_transforms(image_set), return_masks=args.masks)
    return dataset

def build(image_set, args, fpath = None):

    root = osp.join(config.imgDir, '..', 'annotations')
    assert osp.exists(root), f'provided COCO path {root} does not exist'

    PATHS = {
        'train': (config.imgDir, config.train_json),
        'val': (config.imgDir, config.eval_json)
    }
    # if fpath is not None and osp.exists(fpath):
    #     PATHS = {'train':(config.imgDir, config.train_json),
    #              'val':(config.imgDir, fpath)}

    img_folder, ann_file = PATHS[image_set]
    
    dataset = CocoDetection(img_folder, ann_file, phase = image_set,
        transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
