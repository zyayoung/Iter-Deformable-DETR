import os, sys
import numpy as np
import torch.nn.functional as F
from torch import nn
from config import config
from box_ops import box_cxcywh_to_xyxy, box_iou
import torch, pdb
class MHAttention(nn.Module):
    """The full multihead attention block"""
    def __init__(self, d_model, n_heads, dropout = 0.):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        assert self.d_model % self.n_heads == 0
        
        self.q = nn.Conv1d(d_model, d_model, 1, stride=1, groups=n_heads, bias=False)
        self.k = nn.Conv1d(d_model, d_model, 1, stride=1, groups=n_heads, bias=False)
        self.v = nn.Conv1d(d_model, d_model, 1, stride=1, groups=n_heads, bias=False)
        
        self.proj = nn.Conv1d(d_model, d_model, 1, stride=1, groups = 1)
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()

    def _reset_parameters(self,):

        for p in [self.k, self.q, self.v, self.proj]:

            if isinstance(p, nn.Conv1d):
                nn.init.xavier_uniform_(p.weight)

        nn.init.constant_(self.proj.bias, 0)
    
    def forward(self, queries, keys, values, mask):

        mask = mask.transpose(1, 2)
        bs, c, num_queries = queries.shape
        d_k = c // self.n_heads
        
        q = self.q(queries)
        k = self.k(keys)
        v = self.v(values)

        q, k = q * mask, k * mask
        q, k = q.reshape(bs, -1, d_k, num_queries), k.reshape(bs, -1, d_k, num_queries)
        v = v.reshape(bs, -1, d_k, num_queries)
        
        attn = torch.einsum('nmcd, nmce -> nmde', q, k)
        
        if self.dropout is not None:
            p_attn = self.dropout(attn)
        
        atn = self._softmax(p_attn, mask[:, :, None, :])
        value = torch.einsum('nmkd, nmcd -> nmck', atn, v *  mask.unsqueeze(2))
        value = value.reshape(bs, -1, num_queries)
        out = self.proj(value) 

        return (out * mask).transpose(1, 2)

    
    def _softmax(self, logits, mask, dim = -1):
        
        eps = 1e-8
        max_value = (logits * mask).max(dim=dim, keepdims = True)[0].detach()
        logits = (logits - max_value) * mask
        
        m = torch.exp(logits) * mask
        n = (torch.exp(logits) * mask).sum(dim=dim, keepdims = True)
        
        p_mask = mask * mask.transpose(2, 3)
        p_val = m / (n + eps)

        return p_mask * p_val
    
class LocalMHAttention(nn.Module):
    """The full multihead attention block"""
    def __init__(self, d_model, n_heads, dropout = 0.):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        assert self.d_model % self.n_heads == 0
        
        self.q = nn.Conv1d(d_model, d_model, 1, stride=1, groups=n_heads, bias=False)
        self.k = nn.Conv1d(d_model, d_model, 1, stride=1, groups=n_heads, bias=False)
        self.v = nn.Conv1d(d_model, d_model, 1, stride=1, groups=n_heads, bias=False)
        
        self.proj = nn.Conv1d(d_model, d_model, 1, stride=1, groups = 1)
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()

    def _reset_parameters(self,):

        for p in [self.k, self.q, self.v, self.proj]:

            if isinstance(p, nn.Conv1d):
                nn.init.xavier_uniform_(p.weight)

        nn.init.constant_(self.proj.bias, 0)
    
    def forward(self, queries, keys, values, boxes, cur_mask):

        bs, c, num_queries = queries.shape
        d_k = c // self.n_heads
        
        q = self.q(queries)
        k = self.k(keys)
        v = self.v(values)
        
        pred_boxes = boxes.reshape(-1, boxes.shape[-1])
        pred_boxes = box_cxcywh_to_xyxy(pred_boxes)

        overlaps, _ = box_iou(pred_boxes, pred_boxes)

        ious = overlaps.reshape(bs, num_queries, bs, num_queries)
        ious = torch.stack([ious[i, :, i, :] for i in range(bs)], dim = 0)

        ious = ious * cur_mask * cur_mask.transpose(1, 2)
        mask = (ious > config.iou_thr).float().detach()

        q, k = q.reshape(bs, -1, d_k, num_queries), k.reshape(bs, -1, d_k, num_queries)
        v = v.reshape(bs, -1, d_k, num_queries)
        
        attn = torch.einsum('nmcd, nmce -> nmde', q, k)
        
        # 问题来了,周围没有邻居框,周围有至少一个邻居框应该怎样进行self-attention?
        has_neighbor = (mask.sum(dim = -1, keepdim=True) > 0).float().permute(0, 2, 1)
        if self.dropout is not None:
            p_attn = self.dropout(attn)
        
        atn = self._softmax(p_attn, mask.unsqueeze(1))

        #如果当前框没有邻居框,则特征直接传递,如果周围有邻居,则需要进行self-attention.        
        value = torch.einsum('nmkd, nmcd -> nmck', atn, v)
        value = value.reshape(bs, -1, num_queries)
        
        value = value * has_neighbor + (1 - has_neighbor) * v.reshape(bs, -1, num_queries)
        out = self.proj(value)

        return out

    def _softmax(self, logits, mask, dim = -1):
        
        eps = 1e-8
        max_value = (logits * mask).max(dim=dim, keepdims = True)[0].detach()
        logits = (logits - max_value) * mask
        
        m = torch.exp(logits) * mask
        n = (torch.exp(logits) * mask).sum(dim=dim, keepdims = True)
        
        p_val = m / (n + eps)

        return p_val
