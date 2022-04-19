import os,sys
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from config import config
from misc import inverse_sigmoid
from box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou
import torch, pdb

class RelationNet(nn.Module):

    def __init__(self, d_model, activation = 'relu', neighbors = 10):
        
        super().__init__()

        assert neighbors > 0
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)

        # self.fc3 = nn.Linear(2*d_model + 64, d_model)
        self.linear3 = nn.Linear(2 * d_model + 64, d_model)
        self.linear4 = nn.Linear(d_model, d_model)

        self.linear5 = nn.Linear(d_model, d_model)

        self.activation = _get_activation_fn(activation)
        self.top_k = neighbors

    def _recover_boxes(self, container):

        assert np.logical_and('reference_points' in container, 'pred_boxes' in container)

        reference_points = container['reference_points']
        pred_boxes = container['pred_boxes']
        
        if reference_points.shape[-1] == 4:
            new_reference_points = pred_boxes + inverse_sigmoid(reference_points)
            new_reference_points = new_reference_points.sigmoid()
        else:
            assert reference_points.shape[-1] == 2
            new_reference_points[..., :2] = pred_boxes[..., :2] + inverse_sigmoid(reference_points)
            new_reference_points = new_reference_points.sigmoid()
        
        tmp = new_reference_points.detach()
        
        return tmp

    @torch.no_grad()
    def sin_cos_encoder(self, boxes, indices):
        # 这一部分的主要工作是对detected boxes之间的关系进行sin/cos编码.
        eps = 1e-7
        cur_boxes = boxes.unsqueeze(2)
        neighbors = torch.gather(boxes.unsqueeze(1).repeat_interleave(1000, 1), 2, indices[..., :4])

        delta_ctrs = neighbors - cur_boxes
        position_mat = torch.log(torch.clamp(torch.abs(delta_ctrs), eps))
        return self._extract_position_embedding(position_mat)
    
    def _extract_position_embedding(self, position_mat, num_pos_feats=64,temperature=1000):

        num_pos_feats = 128
        temperature = 10000
        scale = 2 * np.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device = position_mat.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_mat = scale * position_mat.unsqueeze(-1) / dim_t.reshape(1, 1, 1, 1, -1)
        pos = torch.stack((pos_mat[:, :, :, 0::2].sin(), pos_mat[:, :, :, 1::2].cos()), dim=4).flatten(3)

        return pos

    def forward(self, tgt, seed_mask, pred_boxes):
        bs, num_queries = pred_boxes.shape[:2]
        
        # recover the predicted boxes
        box_xyxy = box_cxcywh_to_xyxy(pred_boxes)

        # compute the overlaps between boxes and reference boxes
        ious = torch.stack([box_iou(boxes, boxes)[0] for boxes in box_xyxy])

        attn_mask = ious >= config.iou_thr
        
        # use masking to mask the overlap
        neg_mask = (1 - seed_mask)
        overlaps = ious * seed_mask.permute(0, 2, 1) * neg_mask

        c = tgt.shape[-1]
        indices = torch.argsort(-overlaps)[..., :self.top_k].unsqueeze(-1).repeat_interleave(c, dim = -1)
        
        nmask = seed_mask.permute(0, 2, 1).repeat_interleave(num_queries, dim = 1)
        nmk = torch.gather(nmask, 2, indices[..., 0])
        ious = torch.gather(overlaps, 2 ,indices[..., 0])

        mk = nmk * (ious >= config.iou_thr)
        overs = (ious * mk).unsqueeze(-1).repeat_interleave(64, dim = -1)
        waves = self.sin_cos_encoder(pred_boxes, indices)
        
        wave_features = waves
        features = torch.cat([overs, wave_features], dim = -1)
   
        cur = self.linear2(self.activation(self.linear1(tgt)))
        features = self.linear4(self.activation(self.linear3(features)))
        
        # mask the features.
        cur_tgt = cur * neg_mask + (features * mk.unsqueeze(-1)).max(dim = 2)[0]
        
        # update feature of target
        cur_tgt = self.activation(self.linear5(cur_tgt)) * neg_mask

        return cur_tgt, attn_mask

def _get_activation_fn(activation):

    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

# 这一模块在decoder不同stage中进行共享.    
def build_relation(args):

    return RelationNet(d_model = args.hidden_dim, activation = 'relu', neighbors = 10)

def build_relation_net(d_model=256, activation='relu', neighbors = 10):

    return RelationNet(d_model, activation, neighbors)
