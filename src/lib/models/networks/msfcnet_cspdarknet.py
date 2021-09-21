import torch
import torch.nn as nn
from torch.nn import functional as F

from models.n_utils.task_head import head
from models.n_utils.Bifpn import BiFPN
from models.n_utils.parallel_3dconv_v2 import Parallel_conv
from models.n_utils.featureAlign import FeatureAlign


class Mish(nn.Module):
    def __init__(self):
        super(Mish,self).__init__()

    def forward(self,x):
        return x * torch.tanh(F.softplus(x))


class ConvBNMish(nn.Module):

    def __init__(self,inchannels,outchannels,kernel_size,stride=1):
        super(ConvBNMish,self).__init__()
        self.conv = nn.Conv2d(inchannels,outchannels,kernel_size=kernel_size,stride=stride,padding=kernel_size//2,bias=False)
        self.bn = nn.BatchNorm2d(outchannels)  # default momentum = 0.1
        self.activation  = Mish()



    def forward(self,x):
        return self.activation(self.bn(self.conv(x)))



# Residual Block
class ResBlock(nn.Module):
    def __init__(self,in_channels,hidden_channels=None):
        super(ResBlock,self).__init__()

        if hidden_channels is None:
            hidden_channels = in_channels

        self.block = nn.Sequential(ConvBNMish(in_channels,hidden_channels,kernel_size=1),
                                   ConvBNMish(hidden_channels,in_channels,kernel_size=3))


    def forward(self , x):
        return x + self.block(x)


#CSPDarknet block

class CSPblock(nn.Module):
    def __init__(self,in_channels,out_channels,num_block,first):
        super(CSPblock,self).__init__()

        self.downsample = ConvBNMish(in_channels,out_channels,kernel_size=3,stride=2)

        if first:

            self.split_conv0 = ConvBNMish(out_channels,out_channels,kernel_size=1)
            self.split_conv1 = ConvBNMish(out_channels,out_channels,kernel_size=1)

            self.blocks_conv = nn.Sequential(
                ResBlock(in_channels=out_channels,hidden_channels=out_channels//2),
                ConvBNMish(out_channels,out_channels,kernel_size=1)
            )
            self.concat_conv = ConvBNMish(out_channels*2,out_channels,1)


        else:
            self.split_conv0 = ConvBNMish(out_channels,out_channels//2,1)
            self.split_conv1 = ConvBNMish(out_channels,out_channels//2,1)
            self.blocks_conv = nn.Sequential(
                *[ResBlock(out_channels //2 ) for _ in range(num_block)],
                ConvBNMish(out_channels //2, out_channels//2, kernel_size=1)
            )

            self.concat_conv = ConvBNMish(out_channels,out_channels,1)


    def forward(self,x):
        x = self.downsample(x)

        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)

        out = torch.cat([x1,x0],dim=1)
        out = self.concat_conv(out)

        return out





# CSPDarknet53 backbone
class CSPDarknet53(nn.Module):
    def __init__(self,layer_num,heads,head_conv):
        super(CSPDarknet53,self).__init__()

        self.in_channels = 32

        self.conv1 = ConvBNMish(3,self.in_channels,kernel_size=3,stride=1)

        filters = [64,128,256,512,1024]

        self.stages = nn.ModuleList(
            [
                CSPblock(self.in_channels,filters[0],layer_num[0],first=True),
                CSPblock(filters[0],filters[1],layer_num[1],first=False),
                CSPblock(filters[1],filters[2],layer_num[2],first=False),
                CSPblock(filters[2], filters[3], layer_num[3], first=False),
                CSPblock(filters[3], filters[4], layer_num[4], first=False)

        ])

        # --- Before bifpn, channel cut -------------
        self.cut_channel_p3 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_p3 = nn.BatchNorm2d(128)
        self.cut_channel_p4 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, groups=128, bias=False)
        self.bn_p4 = nn.BatchNorm2d(128)
        self.cut_channel_p5 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, groups=128, bias=False)
        self.bn_p5 = nn.BatchNorm2d(128)
        self.cut_channel_p6 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, groups=128, bias=False)
        self.bn_p6 = nn.BatchNorm2d(128)
        self.cut_channel_p7 = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0, groups=128, bias=False)
        self.bn_p7 = nn.BatchNorm2d(128)

        self.relu = nn.ReLU()

        self.bifpn = BiFPN(128)
        self.para_conv = Parallel_conv(pconv_deform=True)

        self.conv3_out = FeatureAlign(128)
        self.conv5_out = FeatureAlign(128)
        self.conv7_out = FeatureAlign(128)

        self.conv_downsampling = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.channel_add128 = nn.Conv2d(128, 256, kernel_size=1, stride=1)
        self.channel_add256 = nn.Conv2d(128, 256, kernel_size=1, stride=1)

        self.fractal_conv_12 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=(1, 2))
        self.fractal_conv_21 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=(2, 1))

        self.head = head(heads, head_conv)


    def forward(self,input):

        x = self.conv1(input)      #32x512x512

        p3 = self.stages[0](x)      #64x256x256
        p4 = self.stages[1](p3)      #128x128x128
        p5 = self.stages[2](p4)      #256x64x64
        p6 = self.stages[3](p5)      #512x32x32
        p7 = self.stages[4](p6)      #1024x16x16


        p3 = self.relu(self.bn_p3(self.cut_channel_p3(p3)))
        p4 = self.relu(self.bn_p4(self.cut_channel_p4(p4)))
        p5 = self.relu(self.bn_p5(self.cut_channel_p5(p5)))
        p6 = self.relu(self.bn_p6(self.cut_channel_p6(p6)))
        p7 = self.relu(self.bn_p7(self.cut_channel_p7(p7)))


        p3_bout, p4_bout, p5_bout, p6_bout, p7_bout = self.bifpn((p3, p4, p5, p6, p7))
        [p3_out, p5_out, p7_out] = self.para_conv([p3_bout, p4_bout, p5_bout, p6_bout, p7_bout])


        # p4_attention
        p3_out = self.conv3_out(p3, p3_out)
        # p5_attention
        p5_out = self.conv5_out(p5, p5_out)
        # p6_attention
        p7_out = self.conv7_out(p7, p7_out)



        p7_128 = F.interpolate(p7_out, scale_factor=8, mode='bilinear', align_corners=True)
        p5_128 = F.interpolate(p5_out, scale_factor=2, mode='bilinear', align_corners=True)
        p3_128 = self.conv_downsampling(p3_out)
        p128 = p3_128 + p5_128 + p7_128

        p128 = self.channel_add128(p128)

        p128x64 = self.fractal_conv_12(p128)
        p128x32 = self.fractal_conv_12(p128x64)

        p64x128 = self.fractal_conv_21(p128)
        p32x128 = self.fractal_conv_21(p64x128)

        p7_256 = F.interpolate(p7_out, scale_factor=16, mode='bilinear', align_corners=True)
        p5_256 = F.interpolate(p5_out, scale_factor=4, mode='bilinear', align_corners=True)
        p256 = p3_out + p5_256 + p7_256

        p256 = self.channel_add256(p256)

        ret = []

        ret.append(self.head(p256))
        ret.append(self.head(p128))
        ret.append(self.head(p128x32))
        ret.append(self.head(p32x128))

        return ret





def get_msfcnetcspdarknet(num_layers ,heads, head_conv):
    #print(type(num_layers))
    if num_layers == 53:
        layer_list = [1,2,8,8,4]
    else:
        layer_list = []
    model = CSPDarknet53(layer_list,heads,head_conv)

    return model









