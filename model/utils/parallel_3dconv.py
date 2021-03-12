import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.branch_conv import branch_conv
from torch.nn import init as init



# 3-D parallel convolution module

class Parallel_conv(nn.Module):
    def __init__(self,
                 in_channels =[128] * 5,
                 out_channels = 128,
                 num_outs = 5,
                 pconv_deform = False,
                 iBN = False,
                 Pconv_num = 1):


        super(Parallel_conv, self).__init__()

        assert isinstance(in_channels,list)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        assert num_outs == 5

        self.iBN = iBN
        self.Pconvs = nn.ModuleList()

        for i in range(Pconv_num):
            self.Pconvs.append(PConvModule(in_channels[i], out_channels, iBN=self.iBN, part_deform=pconv_deform))


    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        x = inputs
        for pconv in self.Pconvs:
            x = pconv(x)
        return x






class PConvModule(nn.Module):
    def __init__(self,
                 in_channels = 256,
                 out_channels = 256,
                 kernel_size = [3, 3, 3],
                 dilation = [1, 1, 1],
                 groups = [1, 1, 1],
                 iBN = False,
                 part_deform = False):

        super(PConvModule,self).__init__()
        print(out_channels)

        # assert not (bias and iBN)

        self.iBN = iBN
        self.Pconv = nn.ModuleList()

        self.Pconv.append(
            branch_conv(in_channels, out_channels, kernel_size=kernel_size[0], dilation=dilation[0], groups = groups[0],
                      padding=(kernel_size[0] + (dilation[0] - 1) * 2) // 2,part_deform=part_deform))
        self.Pconv.append(
            branch_conv(in_channels, out_channels, kernel_size=kernel_size[1], dilation=dilation[1], groups=groups[1],
                      padding=(kernel_size[1] + (dilation[1] - 1) * 2) // 2, part_deform=part_deform))
        self.Pconv.append(
            branch_conv(in_channels, out_channels, kernel_size=kernel_size[2], dilation=dilation[2], groups=groups[2],
                      padding=(kernel_size[2] + (dilation[2] - 1) * 2) // 2, stride = 2, part_deform=part_deform))

        if self.iBN:
            self.bn = nn.BatchNorm2d(128)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)



        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for m in self.Pconv:
            init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x):
        next_x = []
        for level, feature in enumerate(x):
            if level == 1 or level == 3:
                continue
            temp_fea = self.Pconv[1](0 , feature)
            if level > 0:
                temp_fea += self.Pconv[2](0, x[level -1])
            if level < len(x) - 1:
                temp_fea += F.interpolate(self.Pconv[0](1 , x[level + 1]),
                                                size = [temp_fea.size(2),temp_fea.size(3)],mode='bilinear',align_corners=True)

            next_x.append(temp_fea)

        if self.iBN:
            next_x = iBN(next_x,self.bn)

        next_x[0] = self.bn1(next_x[0])
        next_x[1] = self.bn2(next_x[1])
        next_x[2] = self.bn3(next_x[2])


        next_x = [self.relu(item) for item in next_x]

        return next_x


def iBN(fms, bn):
    sizes = [p.shape[2:] for p in fms]
    n,c = fms[0].shape[0] , fms[0].shape[1]

    fm = torch.cat([p.view(n,c,1,-1) for p in fms], dim = -1)
    fm = bn(fm)
    fm = torch.split(fm, [s[0] * s[1] for s in sizes], dim = -1)
    return [p.view(n, c, s[0], s[1]) for p ,s in zip(fm, sizes)]
