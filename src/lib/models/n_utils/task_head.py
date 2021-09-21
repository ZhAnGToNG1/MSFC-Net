import torch.nn as nn
from models.n_utils.featuremap_vis import vis_feature

class head(nn.Module):
    def __init__(self , heads ,head_conv):
        super(head,self).__init__()
        self.heads = heads

        self.relu = nn.ReLU(inplace=True)

        # hm- prediction
        self.hm_64 = nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True)
        self.hm_out = nn.Conv2d(head_conv, self.heads['hm'], kernel_size=1, stride=1, padding=0)

        # wh -regression
        self.wh_64 = nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True)
        self.wh_out = nn.Conv2d(head_conv, self.heads['wh'], kernel_size=1, stride=1, padding=0)

        # reg-regression
        self.reg_64 = nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True)
        self.reg_out = nn.Conv2d(head_conv, self.heads['reg'], kernel_size=1, stride=1, padding=0)

        self.init_weight()




    def init_weight(self):
        print('head_net init weight...')
        nn.init.kaiming_normal_(self.hm_out.weight)
        nn.init.constant_(self.hm_out.bias, -2.19)

        nn.init.kaiming_normal_(self.wh_out.weight)
        nn.init.constant_(self.wh_out.bias, 0)

        nn.init.kaiming_normal_(self.reg_out.weight)
        nn.init.constant_(self.reg_out.bias, 0)


    def forward(self, x):

        ret = {}
        hm_64 = self.hm_64(x)
        hm = self.relu(hm_64)
        hm = self.hm_out(hm)
        ret['hm'] = hm

        # wh-regression
        wh_64 = self.wh_64(x)
        wh = self.relu(wh_64)
        wh = self.wh_out(wh)
        ret['wh'] = wh

        # reg-regression
        reg_64 = self.reg_64(x)
        reg = self.relu(reg_64)
        reg = self.reg_out(reg)
        ret['reg'] = reg

        return ret