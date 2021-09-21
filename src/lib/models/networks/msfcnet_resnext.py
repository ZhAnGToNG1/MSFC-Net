"""ResNet variants"""
import torch
import torch.nn.functional as F
from torch import Tensor


from models.n_utils.task_head import head
from models.n_utils.Bifpn import BiFPN
from models.n_utils.parallel_3dconv_v2 import Parallel_conv
from models.n_utils.featureAlign import FeatureAlign
from typing import Callable, Optional

import torch.nn as nn


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3_xt(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 32,
        base_width: int = 8,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3_xt(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_xt(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride= 1,
        downsample= None,
        groups= 1,
        base_width = 64,
        dilation =1,
    ):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3_xt(width, width, stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out





class ResNeXt(nn.Module):

    def __init__(self, block, layers, heads,head_conv, groups=32, width_per_group=8, model_name = 'resnext101_32x8d'):
        super(ResNeXt, self).__init__()
        self.inplanes = 64
        self.groups = groups
        self.dilation = 1
        self.base_width = width_per_group
        self.model_name = model_name




        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)


        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,dilate = False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,dilate = False)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,dilate = False)






        #--- Before bifpn, channel cut -------------
        self.cut_channel_p4 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, groups=128, bias=False)
        self.bn_p4 = nn.BatchNorm2d(128)
        self.cut_channel_p5 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, groups=128, bias=False)
        self.bn_p5 = nn.BatchNorm2d(128)
        self.cut_channel_p6 = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0, groups=128, bias=False)
        self.bn_p6 = nn.BatchNorm2d(128)
        self.cut_channel_p7 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, groups=128, bias=False)
        self.bn_p7 = nn.BatchNorm2d(128)

        self.expend_conv1 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.expend_bn1 = nn.BatchNorm2d(128)

        self.bifpn = BiFPN(128)
        self.para_conv = Parallel_conv(pconv_deform = True)

        self.conv3_out = FeatureAlign(128)
        self.conv5_out = FeatureAlign(128)
        self.conv7_out = FeatureAlign(128)

        self.conv_downsampling = nn.Conv2d(128,128,kernel_size = 3, stride=2, padding=1)

        self.channel_add128 = nn.Conv2d(128,256,kernel_size=1,stride=1)
        self.channel_add256 = nn.Conv2d(128,256, kernel_size=1, stride=1)
        self.fractal_conv_12 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=(1, 2))
        self.fractal_conv_21 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=(2, 1))

        self.head = head(heads,head_conv)

    def _make_layer(self, block, planes, blocks, stride=1,dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,self.base_width,previous_dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups = self.groups,base_width=self.base_width,dilation = self.dilation))

        return nn.Sequential(*layers)






    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)      #256x256x128
        p3 = self.relu(self.expend_bn1(self.expend_conv1(x)))  # 256x256x128
        x = self.maxpool(x)   # 128x128x128


        l1 = self.layer1(x)   #128x128x256
        l2 = self.layer2(l1)  #64x64x512
        l3 = self.layer3(l2)  #32x32x1024
        l4 = self.layer4(l3)  #16x16x2048


        p4 = self.cut_channel_p4(l1)  # 128x128x128
        p4 = self.bn_p4(p4)
        p4 = self.relu(p4)

        p5 = self.cut_channel_p5(l2)  # 64x64x128
        p5 = self.bn_p5(p5)
        p5 = self.relu(p5)

        p6 = self.cut_channel_p6(l3)  # 32x32x128
        p6 = self.bn_p6(p6)
        p6 = self.relu(p6)

        p7 = self.cut_channel_p7(l4)  # 16x16x128
        p7 = self.bn_p7(p7)
        p7 = self.relu(p7)


        p3_bout, p4_bout, p5_bout, p6_bout, p7_bout = self.bifpn((p3, p4, p5, p6, p7))

        [p3_out, p5_out, p7_out] = self.para_conv([p3_bout, p4_bout, p5_bout, p6_bout, p7_bout]) # 256x256x128  64x64x128  16x16x128

        p3_out = self.conv3_out(p3, p3_out)


        p5_out = self.conv5_out(p5, p5_out)

        # p6_attention
        p7_out = self.conv7_out(p7, p7_out)



        p7_128 = F.interpolate(p7_out,scale_factor=8,mode='bilinear',align_corners=True)
        p5_128 = F.interpolate(p5_out,scale_factor=2,mode='bilinear',align_corners=True)
        p3_128 = self.conv_downsampling(p3_out)
        p128 = p3_128 + p5_128 + p7_128

        p128 = self.channel_add128(p128)

        p128x64 = self.fractal_conv_12(p128)
        p128x32 = self.fractal_conv_12(p128x64)

        p64x128 = self.fractal_conv_21(p128)
        p32x128 = self.fractal_conv_21(p64x128)



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

    def init_weights(self,pretrained=True):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained:
            print('=> load ResNeXt pre-trained model weights')
            save_mode = torch.hub.load_state_dict_from_url(model_urls[self.model_name],progress=True)
            model_dict = self.state_dict()
            state_dict = {k: v for k, v in save_mode.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.load_state_dict(model_dict)

            # self.load_state_dict(torch.hub.load_state_dict_from_url(
            #     resnest_model_urls['resnest101'], progress=True, check_hash=True))



resnet_spec = {50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

def get_msfcnetresnext(num_layers, heads, head_conv):
    if num_layers == 101:
        model_name = 'resnext101_32x8d'
        width_per_group = 8
    elif num_layers == 50:
        model_name = 'resnext50_32x4d'
        width_per_group = 4
    block_class, layers = resnet_spec[num_layers]
    model = ResNeXt(block_class, layers,heads,head_conv=head_conv,groups=32, width_per_group=width_per_group,model_name = model_name)
    model.init_weights(pretrained=True)
    return model
