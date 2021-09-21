import torch.nn as nn
import torch.nn.functional as F
from models.n_utils.task_head import head
from models.utils import conv_3x3
from models.n_utils.Bifpn import BiFPN
from models.n_utils.parallel_3dconv_v2 import Parallel_conv
from models.n_utils.featureAlign import FeatureAlign
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Dilated_bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Dilated_bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)


        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=2, bias=False,dilation=2)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)


        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)


        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Dilated_Bottleneck_projection(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Dilated_Bottleneck_projection, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=2, bias=False, dilation=2)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)

        self.conv1x1 = nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn1x1 = nn.BatchNorm2d(planes*self.expansion, momentum=BN_MOMENTUM)


        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)


        pro = self.conv1x1(x)
        pro = self.bn1x1(pro)

        if self.downsample is not None:
            pro = self.downsample(pro)

        out += pro
        out = self.relu(out)

        return out

class DetNet(nn.Module):
    def __init__(self, block, layers, heads,head_conv,**kwargs):

        self.deconv_with_bias = False
        self.heads = heads
        self.inplanes = 64
        super(DetNet, self).__init__()


        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,    # x2  1
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.expend_conv1 = nn.Conv2d(64,128,kernel_size=1,stride=1,padding=0)
        self.expend_bn1 = nn.BatchNorm2d(128,momentum=BN_MOMENTUM)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  #x
        self.layer1 = self._make_layer(block, 64, layers[0])              #x4     2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)   #x8     3
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)   #x16    4
        self.layer4 = self._make_detnet_layer(256, layers[3], stride=1)   #x16    5
        self.layer5 = self._make_detnet_layer(256, layers[4], stride=1)   #X16    6
        self.layer6_conv3x3 = nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=2,padding=1)

        #--- Before bifpn, channel cut -------------
        self.cut_channel_p4 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, groups=128, bias=False)
        self.bn_p4 = nn.BatchNorm2d(128)
        self.cut_channel_p5 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, groups=128, bias=False)
        self.bn_p5 = nn.BatchNorm2d(128)
        self.cut_channel_p6 = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0, groups=128, bias=False)
        self.bn_p6 = nn.BatchNorm2d(128)
        self.cut_channel_p7 = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0, groups=128, bias=False)
        self.bn_p7 = nn.BatchNorm2d(128)
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




    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            #add data dim 64->256
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_detnet_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes *4:
            #add data dim 64->256
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Dilated_Bottleneck_projection.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Dilated_Bottleneck_projection.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(Dilated_Bottleneck_projection(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Dilated_Bottleneck_projection.expansion
        for i in range(1, blocks):
            layers.append(Dilated_bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):

        # Backbone network
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)      #256x256x64
        p3 = self.relu(self.expend_bn1(self.expend_conv1(x)))    #256x256x128
        x = self.maxpool(x)   # 128x128x128


        l1 = self.layer1(x)   #128x128x256
        l2 = self.layer2(l1)  #64x64x512
        l3 = self.layer3(l2)  #32x32x1024
        l4 = self.layer4(l3)  #32x32x1024
        l5 = self.layer5(l4)  #32x32x1024

        l6 = self.relu(self.layer6_conv3x3(l5))



        p4 = self.cut_channel_p4(l1)  # 128x128x128
        p4 = self.bn_p4(p4)
        p4 = self.relu(p4)

        p5 = self.cut_channel_p5(l2)  # 64x64x128
        p5 = self.bn_p5(p5)
        p5 = self.relu(p5)

        p6 = self.cut_channel_p6(l5)  # 32x32x128
        p6 = self.bn_p6(p6)
        p6 = self.relu(p6)

        p7 = self.cut_channel_p7(l6)  # 16x16x128
        p7 = self.bn_p7(p7)
        p7 = self.relu(p7)



        # MSSF module
        p3_bout, p4_bout, p5_bout, p6_bout, p7_bout = self.bifpn((p3, p4, p5, p6, p7))
        [p3_out, p5_out, p7_out] = self.para_conv([p3_bout, p4_bout, p5_bout, p6_bout, p7_bout]) # 256x256x128  64x64x128  16x16x128
        # p4_attention
        p3_out = self.conv3_out(p3, p3_out)
        # p5_attention
        p5_out = self.conv5_out(p5, p5_out)
        # p6_attention
        p7_out = self.conv7_out(p7, p7_out)



        # Fractal convolution regression layers
        # scale1,2,3:   128x128  128x32  32x128
        p7_128 = F.interpolate(p7_out,scale_factor=8,mode='bilinear',align_corners=True)
        p5_128 = F.interpolate(p5_out,scale_factor=2,mode='bilinear',align_corners=True)
        p3_128 = self.conv_downsampling(p3_out)
        p128 = p3_128 + p5_128 + p7_128
        p128 = self.channel_add128(p128)

        p128x64 = self.fractal_conv_12(p128)
        p128x32 = self.fractal_conv_12(p128x64)

        p64x128 = self.fractal_conv_21(p128)
        p32x128 = self.fractal_conv_21(p64x128)


        # scale4: 256x256
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

    def init_weights(self,num_layers,pretrained=True):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained:
            print('=> load ResNet pre-trained model weights')
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))

            model_dict = self.state_dict()
            model_stat = {}

            for k,v in pretrained_state_dict.items():
                if 'layer4' in k or 'fc' in k:
                    continue
                else:
                    model_stat[k] = v



            model_dict.update(model_stat)
            self.load_state_dict(model_dict)


resnet_spec = {50: (Bottleneck, [3, 4, 6, 3, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

def get_msfcnetdetnet(num_layers, heads, head_conv):
    block_class, layers = resnet_spec[num_layers]
    model = DetNet(block_class, layers,heads,head_conv=head_conv)
    model.init_weights(num_layers,pretrained=True)
    return model
