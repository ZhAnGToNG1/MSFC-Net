"""ResNet variants"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.n_utils.splat import SplAtConv2d
from models.n_utils.task_head import head
from models.n_utils.Bifpn import BiFPN
from models.n_utils.parallel_3dconv import Parallel_conv
from models.n_utils.featureAlign import FeatureAlign



_url_format = 'https://hangzh.s3.amazonaws.com/encoding/models/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('528c19ca', 'resnest50'),
    ('22405ba7', 'resnest101'),
    ('75117900', 'resnest200'),
    ('0cc87c48', 'resnest269'),
    ]}

def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]
resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}



class DropBlock2D(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)

class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 rectified_conv=False, rectify_avg=False,
                 norm_layer=None, dropblock_prob=0.0, last_gamma=False):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)

        if radix > 1:
            self.conv2 = SplAtConv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                radix=radix, rectify=rectified_conv,
                rectify_avg=rectify_avg,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob)
        elif rectified_conv:
            from rfconv import RFConv2d
            self.conv2 = RFConv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False,
                average_mode=rectify_avg)
            self.bn2 = norm_layer(group_width)
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(
            group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 1:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, heads,head_conv, radix=1, groups=1, bottleneck_width=64
                 , dilated=False, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False,
                 rectified_conv=False, rectify_avg=False,
                 avd=False, avd_first=False, dropblock_prob=0,
                 last_gamma=False, norm_layer=nn.BatchNorm2d):

        self.deconv_with_bias = False
        self.heads = heads

        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width*2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        super(ResNet, self).__init__()
        self.rectified_conv = rectified_conv
        self.rectify_avg = rectify_avg
        if rectified_conv:
            from rfconv import RFConv2d
            conv_layer = RFConv2d
        else:
            conv_layer = nn.Conv2d
        conv_kwargs = {'average_mode': rectify_avg} if rectified_conv else {}

        if deep_stem:
            self.conv1 = nn.Sequential(
                conv_layer(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False, **conv_kwargs),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
            )
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False, **conv_kwargs)


        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        elif dilation==2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilation=1, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)


        #--- Before bifpn, channel cut -------------
        self.cut_channel_p4 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, groups=128, bias=False)
        self.bn_p4 = nn.BatchNorm2d(128)
        self.cut_channel_p5 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, groups=128, bias=False)
        self.bn_p5 = nn.BatchNorm2d(128)
        self.cut_channel_p6 = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0, groups=128, bias=False)
        self.bn_p6 = nn.BatchNorm2d(128)
        self.cut_channel_p7 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, groups=128, bias=False)
        self.bn_p7 = nn.BatchNorm2d(128)


        # ----  BiFpn and 3d_conv module ----------
        self.bifpn = BiFPN(128)
        self.para_conv = Parallel_conv(pconv_deform = True)


        #------ CF(compensatory fusion) module---------
        self.conv3_out = FeatureAlign(128)
        self.conv5_out = FeatureAlign(128)
        self.conv7_out = FeatureAlign(128)


        self.conv_downsampling = nn.Conv2d(128,128,kernel_size = 3, stride=2, padding=1)
        self.channel_add128 = nn.Conv2d(128,256,kernel_size=1,stride=1)
        self.channel_add256 = nn.Conv2d(128,256, kernel_size=1, stride=1)
        #---------FC-------------------
        self.fractal_conv_12 = nn.Conv2d(256, 256,kernel_size=3,padding=1,stride=(1, 2))
        self.fractal_conv_21 = nn.Conv2d(256, 256,kernel_size=3,padding=1,stride=(2, 1))

        self.head = head(heads,head_conv)




    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    dropblock_prob=0.0, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
                                                    ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
                                                    ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=1, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=2, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=dilation, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))

        return nn.Sequential(*layers)






    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)      #256x256x128

        p3 = x
        x = self.maxpool(x)   #128x128x128
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
            print('=> load ResNeSt pre-trained model weights')
            save_model = torch.hub.load_state_dict_from_url(resnest_model_urls['resnest101'], progress=True)
            model_dict = self.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}

            model_dict.update(state_dict)
            self.load_state_dict(model_dict)


resnet_spec = {50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

def get_msfcnet(num_layers, heads, head_conv):
    block_class, layers = resnet_spec[num_layers]
    model = ResNet(block_class, layers,heads,head_conv=head_conv,
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False)
    model.init_weights(pretrained=True)
    return model
