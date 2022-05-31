# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import torch
from common import *
import datasets
import misc as utils
from torch.utils.data import DataLoader
from datasets.coco import construct_dataset
from datasets import get_coco_api_from_dataset
from detr import build_model
import pdb

def get_args_parser():
    
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=512, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=200, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--use_checkpoint', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    parser.add_argument('--end_epoch', default=120, type=int, metavar='N',
                        help='end epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--num_gpus', default = 4, type = int)

    parser.add_argument('--dense_query', default=0, type=int)
    parser.add_argument('--rectified_attention', default=0, type=int)
    
    parser.add_argument('--aps', default=0, type=int)

    return parser

def deplicate(record, thr):

    assert 'scores' in record
    names = [k for (k, v) in record.items()]
    flag = record['scores'] >= thr
    for name in names:
        record[name] = record[name][flag]
    return record    

def processor(result_queue, dataset, device, model_file, args):

    assert type(device) == int and device > -1
    fpath = f'{random.randint(0, 10**9-1):09d}.json'
    with open(fpath, 'w+') as fid:
        fid.write(json.dumps(dataset))

    torch.cuda.set_device(device)
    device = torch.device('cuda:{}'.format(device))
    model, _, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    num = np.random.randint(10000)
    
    dataset_val = construct_dataset(fpath, 'val', args)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
        drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    
    base_ds = None
    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    checkpoint = torch.load(model_file, map_location='cpu')
    missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if utils.is_main_process():
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
 
    model.eval()
    result_list, n = [], len(data_loader)
    # pbar = tqdm(total = n, leave = False, ascii = True)
    counter, thr = 0, 0.05
    for samples, targets, filenames in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        results = [deplicate(r, thr) for r in results]
        
        targets = [{k:v.cpu().numpy() for k, v in t.items()} for t in targets]
        results = [{k:v.cpu().numpy() for k, v in r.items()} for r in results]
        dtboxes = [np.hstack([r['boxes'], r['scores'][:, np.newaxis]]) for r in results]
        dtboxes = [boxes_dump(db) for db in dtboxes]
        res = [{'ID':name, 'dtboxes':db} for name, db in zip(filenames, dtboxes)]
        assert len(res) == 1
        result_queue.put_nowait(res[0])
    os.remove(fpath)
    return result_list

def multi_process(func, data, nr_procs, model_file, *args):

    total = len(data['images'])
    stride = math.ceil(total / nr_procs)
    result_queue = Queue(10000)
    results, procs = [], []
    tqdm.monitor_interval = 0
    pbar = tqdm(total = total, leave = False, ascii = True)
    for i in range(nr_procs):
        start = i*stride
        end = np.min([start+stride,total])
        sample_data = data.copy()
        sample_data['images'] = data['images'][start:end]
        # func(result_queue, sample_data, i, model_file, *args)
        p = Process(target= func,args=(result_queue, sample_data, i, model_file, *args))
        p.start()
        procs.append(p)

    for i in range(total):

        t = result_queue.get()
        if t is None:
            pbar.update(1)
            continue
        results.append(t)
        pbar.update(1)

    for p in procs:
        p.join()
    return results

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    ann_file = '/home/zhenganlin/june/CrowdHuman/annotations/val.json'
    with open(ann_file, 'r') as fid:
        line = fid.readlines()
    record = json.loads(line[0])

    model_dir = config.snapshot_dir
    saveDir = config.eval_dir
    model_dir = os.path.join(config.snapshot_dir, args.output_dir)
    saveDir = os.path.join(config.eval_dir, args.output_dir)
    ensure_dir(saveDir)

    for epoch in range(args.start_epoch, args.end_epoch):

        model_file = osp.join(model_dir, 'checkpoint-{}.pth'.format(epoch))
        if not osp.exists(model_file):
            print(model_file)
            continue

        results = multi_process(processor, record, args.num_gpus, model_file, args)
        file_path = osp.join(saveDir, 'epoch-{}.human'.format(epoch))
        save_results(results, file_path)
    
if __name__ == '__main__':

    main()
