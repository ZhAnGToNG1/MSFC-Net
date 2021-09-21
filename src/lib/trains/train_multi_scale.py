from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss_hm
from models.losses import RegL1Loss, IOUloss
from models.utils import _sigmoid
from .base_trainer import BaseTrainer


class CtdetLoss(torch.nn.Module):
  def __init__(self, opt):
    super(CtdetLoss, self).__init__()
    self.hm_lossfunc = FocalLoss_hm()
    self.reg_lossfunc = RegL1Loss()
    self.wh_lossfunc = RegL1Loss()
    self.iou_lossfunc = IOUloss()
    self.opt = opt

  def forward(self, outputs, batches):
    opt = self.opt
    num = len(outputs)
    hm_loss,wh_loss,off_loss,iou_loss = [0]*num,[0]*num,[0]*num,[0]*num


    for i in range(num):
      output = outputs[i]
      batch = batches[i]
      output['hm'] = _sigmoid(output['hm'])
      hm_loss[i] = self.hm_lossfunc(output['hm'], batch['hm'])
      wh_loss[i] += self.wh_lossfunc(
        output['wh'], batch['reg_mask'],
        batch['ind'], batch['wh'])
      iou_loss[i] += self.iou_lossfunc(
        output['wh'], batch['reg_mask'],
        batch['ind'], batch['wh'])

      if opt.reg_offset and opt.off_weight > 0:
        off_loss[i] += self.reg_lossfunc(output['reg'], batch['reg_mask'],
                             batch['ind'], batch['reg'])

    total_hm = sum(hm_loss) / num
    total_wh = sum(wh_loss) / num
    total_off = sum(off_loss) / num
    total_iou = sum(iou_loss) / num
    loss = opt.hm_weight * total_hm  + opt.wh_weight * total_wh  +  total_iou + \
           opt.off_weight * total_off

    loss_stats = {'loss': loss, 'hm_loss': total_hm, 'wh_loss': total_wh, 'iou_loss': total_iou, 'off_loss': total_off}

    return loss, loss_stats
class CtdetTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(CtdetTrainer, self).__init__(opt, model, optimizer=optimizer)
  
  def _getlosses(self, opt):
    loss_states = ['loss', 'hm_loss','wh_loss','iou_loss','off_loss']
    # add loss
    loss = CtdetLoss(opt)
    print(loss)
    return loss_states, loss