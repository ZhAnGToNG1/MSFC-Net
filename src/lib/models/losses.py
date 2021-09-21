from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _tranpose_and_gather_feat
import torch.nn.functional as F
import numpy as np


def _slow_neg_loss(pred, gt):
    '''focal loss from CornerNet'''
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _neg_loss_hm(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _neg_loss_seg(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    #neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    num_pos = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -= all_loss
    return loss


def _slow_reg_loss(regr, gt_regr, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr = regr[mask]
    gt_regr = gt_regr[mask]

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


def _reg_loss(regr, gt_regr, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


class FocalLoss_seg(nn.Module):

    def __init__(self):
        super(FocalLoss_seg, self).__init__()
        self.neg_loss = _neg_loss_seg

    def forward(self, out, target):
        return self.neg_loss(out, target)



class FocalLoss_hm(nn.Module):

    def __init__(self):
        super(FocalLoss_hm, self).__init__()
        self.neg_loss = _neg_loss_hm

    def forward(self, out, target):
        return self.neg_loss(out, target)

class RegLoss(nn.Module):

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)

        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss

class NormRegL1Loss(nn.Module):
    def __init__(self):
        super(NormRegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        pred = pred / (target + 1e-4)
        target = target * 0 + 1
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class RegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_lfloss(pred * mask, target * mask, reduction='elementwise_mean')
        return loss


class IOUloss(nn.Module):
    def __init__(self):
        super(IOUloss,self).__init__()


    def forward(self, output,mask,ind,target):
        batch,cat,height,width = output.size()

        pred = _tranpose_and_gather_feat(output,ind)


        mask = mask.unsqueeze(2).expand_as(pred).float()
        pred = pred * mask
        cy = (ind / width).int().float()
        cx = (ind % width).int().float()

        cx = cx.unsqueeze(2)
        cy = cy.unsqueeze(2)

        x1_p = cx - pred[:,:,0:1]/2
        y1_p = cy - pred[:,:,1:2]/2
        x2_p = cx + pred[:,:,0:1]/2
        y2_p = cy + pred[:,:,1:2]/2

        target = target * mask
        x1_g = cx - target[:,:,0:1]/2
        y1_g = cy - target[:,:,1:2]/2
        x2_g = cx + target[:,:,0:1]/2
        y2_g = cy + target[:,:,1:2]/2


        Area_p = (x2_p-x1_p)*(y2_p-y1_p)
        Area_g = (x2_g-x1_g)*(y2_g-y1_g)

        x1_inter = torch.max(x1_g,x1_p)
        x2_inter = torch.min(x2_g,x2_p)
        y1_inter = torch.max(y1_g,y1_p)
        y2_inter = torch.min(y2_g,y2_p)


        flag = x2_inter > x1_inter



        inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        inter_ = inter * flag.float()


        x1_enc = torch.min(x1_g,x1_p)
        x2_enc = torch.max(x2_g,x2_p)

        y1_enc = torch.min(y1_g,y1_p)
        y2_enc = torch.max(y2_g,y2_p)


        enc = (x2_enc-x1_enc)*(y2_enc-y1_enc) + 1e-5

        u = Area_g + Area_p - inter_ + 1e-5
        IOU = (inter / u)*flag.float()

        Giou = IOU - (enc - u) / enc * (flag.float() + 1e-5)

        loss = 1 - Giou

        loss = loss * flag.float()
        loss_sum = loss.sum()
        flag_sum = flag.float().sum()
        return loss_sum / (flag_sum + 1e-5)


















