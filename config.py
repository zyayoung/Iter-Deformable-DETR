import os, getpass
import os.path as osp
import numpy as np
import argparse
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


root_dir = '.'
add_path(osp.join(root_dir, 'lib'))
add_path(osp.join(root_dir, 'util'))
add_path(osp.join(root_dir, 'tools'))


class Config:
    imgDir = '/home/zhenganlin/june/CrowdHuman/images'
    
    output_dir = 'output'
    snapshot_dir = osp.join(output_dir, 'model_dump')
    eval_dir = osp.join(output_dir, 'eval_dump')
    
    train_image_folder, val_image_folder = imgDir, imgDir
    train_json = osp.join(imgDir, '..', 'annotations_new', 'crowdhuman_train.json')
    eval_json = osp.join(imgDir, '..', 'annotations_new', 'crowdhuman_val.json')
    anno_file = osp.join(imgDir, '..', 'odformat', 'crowdhuman_val.odgt')
    train_file = osp.join(imgDir, '..', 'odformat', 'crowdhuman_train.odgt')

    ign_thr = 0.7
    score_thr = 0.7
    watershed = 5
    iou_thr = 0.4
    floor_thr = 0.05
    iter_nums = 1

config = Config()
