from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _gather_feat, _tranpose_and_gather_feat



def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(img_size, heats, whs, regs=None, K=100):
    height0 = img_size // 4
    width0 = img_size // 4


    for i in range(len(heats)):
        heat = heats[i]
        wh = whs[i]
        reg = regs[i]

        batch,cat,height,width = heat.size()
        h_scale = height0 / height
        w_scale = width0 / width
        heat = _nms(heat)
        scores, inds, clses, ys, xs = _topk(heat, K=K)

        if reg is not None:
            reg = _tranpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, K, 2)
            xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, K, 1) + 0.5
            ys = ys.view(batch, K, 1) + 0.5

        wh = _tranpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, K, 2)
        clses = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1)

        x1 = (xs - wh[..., 0:1] / 2) * w_scale
        y1 = (ys - wh[..., 1:2] / 2) * h_scale
        x2 = (xs + wh[..., 0:1] / 2) * w_scale
        y2 = (ys + wh[..., 1:2] / 2) * h_scale

        bboxes = torch.cat([x1,y1,x2,y2], dim=2)

        if i == 0:
            detections = torch.cat([bboxes, scores, clses], dim=2)
        else:
            detections = torch.cat([detections , torch.cat([bboxes,scores,clses] ,dim=2)],dim=1)

    return detections