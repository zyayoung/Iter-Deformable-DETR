# ------------------------------------------------------------------------
# Modified by Matthieu Lin
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import box_ops


class SamplingAttention_RA(nn.Module):
    """decoder attn"""
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super(SamplingAttention_RA, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels

        self.pool_resolution = (1, n_points)  # width height
        self.n_points = self.pool_resolution[0] * self.pool_resolution[1]

        self.value_conv = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.output_proj = nn.Linear(d_model, d_model)

        self.sampling_locs = nn.Linear(d_model, n_heads * self.n_points * self.n_levels * 2)
        self.sampling_weight = nn.Linear(d_model , n_heads * self.n_levels * self.n_points)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_locs.weight.data, 0.)
        constant_(self.sampling_locs.bias.data, 0.)

        constant_(self.sampling_weight.weight.data, 0.)
        constant_(self.sampling_weight.bias.data, 0.)

        xavier_uniform_(self.value_conv.weight.data)
        constant_(self.value_conv.bias.data, 0.)

        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)


    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, \
        input_level_start_index, input_padding_mask=None):
    # def forward(self, q, k, key_padding_mask=None, pos_centers=None, spatial_shape=None):
        q = query.transpose(0, 1)
        k = input_flatten.transpose(1, 2)
        key_padding_mask = input_padding_mask
        pos_centers = reference_points
        spatial_shape_ = input_spatial_shapes
        # valid_sizes = None
        # valid_scales = None

        M_ = self.n_heads
        P_ = self.n_points
        F_ = self.n_levels
        N_, C_, S_ = k.shape  # memoy of encoder
        L_ = q.shape[0]

        # [bs, #level, 2] -> [1, nhead*bs, 1, #level, 2]
        # valid_sizes = valid_sizes.view(1, N_, 1, F_, 2).repeat_interleave(M_, 1)
        # valid_scales = 2 * valid_scales.view(1, N_, 1, F_, 2).repeat_interleave(M_, 1)

        value = self.value_conv(k.unsqueeze(-1)).squeeze(-1)
        value = value.masked_fill(key_padding_mask.view(N_, 1, S_), float(0))

        spatial_splits = [H_ * W_ for H_, W_ in spatial_shape_]
        value_list = torch.split(value, spatial_splits, dim=-1)
        value_list = [value_.view(N_ * M_, C_ // M_, H_, W_) for value_, (H_, W_) in zip(value_list, spatial_shape_)]

        weights = self.sampling_weight(q).view(L_, N_ * M_, 1, F_ * P_).softmax(3)
        # [L, bs, C] -> [L, nhead*bs, #key, #level, 2]
        grids = self.sampling_locs(q).view(L_, N_ * M_, P_, F_, 2)

        # [N * nhead, L, #level, 4] -> [L, bs * nhead, 1, #level, 2]
        pos_centers = pos_centers.repeat_interleave(M_, 0).permute(1, 0, 2, 3).unsqueeze(2)

        ##
        # [L, bs * nhead, 1, #level, 2]
        wh = pos_centers[..., 2:]
        # [L, nhead*bs, #key, #level, 2]
        grid_pts = torch.zeros((L_, M_, P_, F_, 2), dtype=weights.dtype, device=weights.device)

        for h_i in range(M_):
            for i in range(self.n_points):
                grid_pts[:, h_i, i, :, 0] = ((i % int(self.pool_resolution[1])) + 0.5) / self.pool_resolution[1]
                grid_pts[:, h_i, i, :, 1] = (h_i  + 0.5 ) / M_

        grid_pts = grid_pts.repeat(1, N_, 1, 1, 1)
        grid_pts *= wh

        # [N * nhead, L, 4] -> [L, bs*nhead, 1, 1, 2]
        boxes_xy = box_ops.box_cxcywh_to_xyxy(pos_centers)[..., :2]

        grids = ( (grids * wh / P_) + boxes_xy + grid_pts) * 2 - 1

        # [L, bs*nhead, #key, #level, 2] -> [#level, bs*nhead, L, #key, 2]
        grids = grids.permute(3, 1, 0, 2, 4)

        samples_value_list = [F.grid_sample(value, grids, mode='bilinear', padding_mode='zeros', align_corners=False)
                              for value, grids in zip(value_list, grids)]

        # [bs*nhead, C / nhead, L, #key*#level]
        samples_value = torch.cat(samples_value_list, -1)
        # [bs*nhead, 1, L, #level*key]
        weights = weights.permute(1, 2, 0, 3)

        # sum all keys on all level [bs*nhead, C / nhead, L] -> [L, N, C]
        output = torch.sum(samples_value * weights, -1).permute(2, 0, 1).view(L_, N_, C_)
        output = self.output_proj(output)

        return output.permute(1, 0, 2)
