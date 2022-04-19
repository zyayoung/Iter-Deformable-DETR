import copy
from typing import Optional, List
import math
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from config import config
from misc import inverse_sigmoid
from .ms_deform_attn import MSDeformAttn, SamplingAttention_RA
from box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from .relation_net import build_relation_net
from .mhattention import MHAttention, LocalMHAttention
import torch, pdb
                        
class SharedDecHead(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,):

        super().__init__()

        # relation net that process detection iteratively.
        self.relation_net = build_relation_net(d_model, activation, neighbors=10)

        # cross attention
        # self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)

        # self attention
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # local self-attention
        self.local_attn = LocalMHAttention(d_model, n_heads, dropout=dropout)
        self.dropout =  nn.Dropout(dropout)
        self.norm4 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def recover_boxes(self, pred_bbox, points):

        reference = inverse_sigmoid(points)
        if reference.shape[-1] == 4:
            pred_bbox += reference
        else:
            assert reference.shape[-1] == 2
            pred_bbox[..., :2] += reference
        
        return pred_bbox.sigmoid()
    
    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes,            \
                level_start_index, src_padding_mask=None,  src_valid_ratios = None,
                container = None,):
        
        level = container['level']
        assert level >= config.watershed
        class_embed, bbox_embed = container['class_embed'], container['bbox_embed']
        
        cur_all_mask = container['next_mask']
        bs, _ = tgt.shape[:2]

        target = {'tgt': tgt * cur_all_mask}
        tgt = self.relation_net(target, container)

        pred_boxes = container['pred_boxes']

        q = k = self.with_pos_embed(tgt, query_pos).transpose(1, 2)
        v = tgt.transpose(1, 2)
        tgt2 = self.local_attn(q, k, v, pred_boxes, cur_all_mask).transpose(1, 2)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt * cur_all_mask)

        # cross attention need also be equipped with mask since only the left queries should be involved in calculation.
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos) * cur_all_mask,
                                reference_points * cur_all_mask.unsqueeze(-1),
                                src, src_spatial_shapes, level_start_index, src_padding_mask)
        
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt * cur_all_mask)
        
        tgt = self.forward_ffn(tgt)

        pred_logits, pred_bbox = class_embed(tgt), bbox_embed(tgt)
            
        points = container['reference_points']
        pred_boxes = self.recover_boxes(pred_bbox, points)
        scores = pred_logits.sigmoid().max(-1, keepdim=True)[0]

        alive_mask = (scores >= config.score_thr).float()
        next_mask = (cur_all_mask - alive_mask) * (scores >= 0.).float()
        gather_mask = container['gather_mask'] + alive_mask
  
        tmp_container = container.copy()
        _cur_all_mask = cur_all_mask.bool().squeeze(-1)
        pred_logits[~_cur_all_mask] = container['pred_logits'].detach()[~_cur_all_mask]
        tmp = {'pred_logits': pred_logits, 'mask': cur_all_mask,'next_mask': next_mask,
               'cur_all_mask': cur_all_mask, 'alive_mask': alive_mask, 'gather_mask':gather_mask}
        
        tmp_container.update(tmp)
        
        return tgt, tmp_container

def _get_activation_fn(activation):
    
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def build_shared_head(d_model, d_ffn=1024, dropout=0.1, activation='relu',
                      n_levels = 4, n_heads = 8, n_points=4, ):

    return SharedDecHead(d_model, d_ffn, dropout, activation, n_levels, n_heads, n_points)