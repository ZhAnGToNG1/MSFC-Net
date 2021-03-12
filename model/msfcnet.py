import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.task_head import head
from utils.fractalconv import conv3x3
from utils.Bifpn import BiFPN
from utils.parallel_3dconv import Parallel_conv
from utils.featureAlign import FeatureAlign
from backbone.resnest import get_ResNeSt




class MSFC(nn.Module):

    def __init__(self, layers,heads,head_conv):
        super(MSFC, self).__init__()

        self.backbone = get_ResNeSt(num_layers=layers)

        #--- Before bifpn, channel cut -------------
        self.cut_channel_p4 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, groups=128, bias=False)
        self.bn_p4 = nn.BatchNorm2d(128)
        self.cut_channel_p5 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, groups=128, bias=False)
        self.bn_p5 = nn.BatchNorm2d(128)
        self.cut_channel_p6 = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0, groups=128, bias=False)
        self.bn_p6 = nn.BatchNorm2d(128)
        self.cut_channel_p7 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, groups=128, bias=False)
        self.bn_p7 = nn.BatchNorm2d(128)


        # ----  BiFpn and sepc module ----------
        self.bifpn = BiFPN(128)
        self.para_conv = Parallel_conv(Pconv_num= 1, pconv_deform = True, iBN = False)

        #------ compensatory fusion opeartion ---------
        self.conv3_out = FeatureAlign(128)
        self.conv5_out = FeatureAlign(128)
        self.conv7_out = FeatureAlign(128)


        self.conv_downsampling = nn.Conv2d(128,128,kernel_size = 3, stride=2, padding=1)

        self.channel_add128 = nn.Conv2d(128,256,kernel_size=1,stride=1)
        self.channel_add256 = nn.Conv2d(128,256, kernel_size=1, stride=1)

        self.fracatal_conv_1x2 = conv3x3(256, 256, padding=1, stride=(1, 2))
        self.fracatal_conv_2x1 = conv3x3(256, 256, padding=1, stride=(2, 1))

        self.head = head(heads,head_conv)






    def forward(self, x):


        p3,p4,p5,p6,p7 = self.backbone(x)


        p4 = self.cut_channel_p4(p4)  # 128x128x128
        p4 = self.bn_p4(p4)
        p4 = self.relu(p4)

        p5 = self.cut_channel_p5(p5)  # 64x64x128
        p5 = self.bn_p5(p5)
        p5 = self.relu(p5)

        p6 = self.cut_channel_p6(p6)  # 32x32x128
        p6 = self.bn_p6(p6)
        p6 = self.relu(p6)

        p7 = self.cut_channel_p7(p7)  # 16x16x128
        p7 = self.bn_p7(p7)
        p7 = self.relu(p7)



        p3_bout, p4_bout, p5_bout, p6_bout, p7_bout = self.bifpn((p3, p4, p5, p6, p7))



        [p3_out, p5_out, p7_out] = self.para_conv([p3_bout, p4_bout, p5_bout, p6_bout, p7_bout]) # 256x256x128  64x64x128  16x16x128



        # p4_attention
        p3_out = self.conv3_out(p3, p3_out)

        # p5_attention
        p5_out = self.conv5_out(p5, p5_out)

        # p6_attention
        p7_out = self.conv7_out(p7, p7_out)



        p7_128 = F.interpolate(p7_out,scale_factor=8,mode='bilinear',align_corners=True)
        p5_128 = F.interpolate(p5_out,scale_factor=2,mode='bilinear',align_corners=True)
        p3_128 = self.conv_downsampling(p3_out)
        p128 = p3_128 + p5_128 + p7_128

        p128 = self.channel_add128(p128)

        p128x64 = self.fracatal_conv_1x2(p128)
        p128x32 = self.fracatal_conv_1x2(p128x64)

        p64x128 = self.fracatal_conv_2x1(p128)
        p32x128 = self.fracatal_conv_2x1(p64x128)



        p7_256 = F.interpolate(p7_out,scale_factor=16,mode='bilinear',align_corners=True)
        p5_256 = F.interpolate(p5_out,scale_factor=4 ,mode='bilinear',align_corners=True)
        p256 = p3_out + p5_256 + p7_256

        p256 = self.channel_add256(p256)


        ret = []

        ret.append(self.head(p256))
        ret.append(self.head(p128))
        ret.append(self.head(p128x32))
        ret.append(self.head(p32x128))



        return ret  # [8,3]




def get_MSFC(num_layers, heads, head_conv):
    model = MSFC(num_layers,heads,head_conv)
    return model