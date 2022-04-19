# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
from config import config
from box_ops import box_cxcywh_to_xyxy, giou_iou


def masked_assignment(c, mask=None) -> list:
    if mask is None:
        return linear_sum_assignment(c)
    c[mask == 0] = 1e12
    iSet, jSet = linear_sum_assignment(c)
    keep = mask[iSet, jSet]
    return iSet[keep], jSet[keep]


def relation_net_assignment(c, alive_mask, cur_all_mask, valid_mask):
    c = c.numpy()
    i_0, j_0 = masked_assignment(c[alive_mask], valid_mask[alive_mask])
    selected = np.zeros(c.shape[1], dtype=bool)
    selected[j_0] = 1
    i_1, j_1 = masked_assignment(c[cur_all_mask][:, ~selected], valid_mask[cur_all_mask][:, ~selected])
    i_1 = np.where(cur_all_mask)[0][i_1]
    j_1 = np.where(~selected)[0][j_1]
    return i_1, j_1


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
            tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
            cost_giou, iou = giou_iou(out_bbox_xyxy, tgt_bbox_xyxy)
            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * -cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            if outputs['mask'] is not None:
                valid_mask = (tgt_bbox_xyxy[None, :, 0] < out_bbox[:, None, 0]) \
                           & (out_bbox[:, None, 0] < tgt_bbox_xyxy[None, :, 2]) \
                           & (tgt_bbox_xyxy[None, :, 1] < out_bbox[:, None, 1]) \
                           & (out_bbox[:, None, 1] < tgt_bbox_xyxy[None, :, 3]) \
                           & (iou > 0)
                valid_mask = valid_mask.view(bs, num_queries, -1).cpu()
                valid_mask = [m[i].bool().numpy() for i, m in enumerate(valid_mask.split(sizes, -1))]
                alive_mask = outputs['mask']['seed_mask'].cpu().numpy()
                cur_all_mask = outputs['mask']['mask'].cpu().numpy()
                indices = [relation_net_assignment(c[i], alive_mask[i], cur_all_mask[i], valid_mask[i]) for i, c in enumerate(C.split(sizes, -1))]
            else:
                indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou)
