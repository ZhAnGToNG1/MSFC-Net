import torch
import torch.nn as nn
import torch.nn.functional as F


from models.n_utils.task_head import head
from models.n_utils.Bifpn import BiFPN
from models.n_utils.parallel_3dconv_v2 import Parallel_conv
from models.n_utils.featureAlign import FeatureAlign

class VGGNet(nn.Module):
    def __init__(self,heads,head_conv,num_layer):
        super(VGGNet,self).__init__()
        print('num_layer',num_layer)
        if num_layer == 19:
            self.block1 = self.make_layer(conv_num=2,inchannel=3,outchannel=64)
            self.block2 = self.make_layer(conv_num=2,inchannel=64,outchannel=128)
            self.block3 = self.make_layer(conv_num=4,inchannel=128,outchannel=256)
            self.block4 = self.make_layer(conv_num=4,inchannel=256,outchannel=512)
            self.block5 = self.make_layer(conv_num=4,inchannel=512,outchannel=512)

        else:
            self.block1 = self.make_layer(conv_num=2, inchannel=3, outchannel=64)
            self.block2 = self.make_layer(conv_num=2, inchannel=64, outchannel=128)
            self.block3 = self.make_layer(conv_num=3, inchannel=128, outchannel=256)
            self.block4 = self.make_layer(conv_num=3, inchannel=256, outchannel=512)
            self.block5 = self.make_layer(conv_num=3, inchannel=512, outchannel=512)



        self.relu = nn.ReLU()
        self.cut_channel_p3 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, groups=64, bias=False)
        self.bn_p3 = nn.BatchNorm2d(128)
        self.cut_channel_p4 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, groups=128, bias=False)
        self.bn_p4 = nn.BatchNorm2d(128)
        self.cut_channel_p5 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, groups=128, bias=False)
        self.bn_p5 = nn.BatchNorm2d(128)
        self.cut_channel_p6 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, groups=128, bias=False)
        self.bn_p6 = nn.BatchNorm2d(128)
        self.cut_channel_p7 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, groups=128, bias=False)
        self.bn_p7 = nn.BatchNorm2d(128)



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


    def make_layer(self,conv_num,inchannel,outchannel):
        layers = []
        while conv_num > 0:
            layers += [nn.Conv2d(inchannel,outchannel,kernel_size=3,padding=1),nn.BatchNorm2d(outchannel),
                       nn.ReLU()]

            inchannel = outchannel
            conv_num-=1
        layers += [nn.MaxPool2d(kernel_size=2,stride=2)]


        return nn.Sequential(*layers)

    def forward(self,x):


        p3 = self.block1(x)    # 256x256x64

        p4 = self.block2(p3)  #128x128x128

        p5 = self.block3(p4)  #64x64x256

        p6 = self.block4(p5)  #32x32x512

        p7 = self.block5(p6)  #16x16x512


        p3 = self.cut_channel_p3(p3)  # 128x128x128
        p3 = self.bn_p3(p3)
        p3 = self.relu(p3)

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

        return ret  # [8,3]


    def init_weight(self):
        pretrain_model = torch.utils.model_zoo.load_url('https://download.pytorch.org/models/vgg19_bn-c79401a0.pth')
        model = self.state_dict()

        model_key = {}
        for k ,v in model.items():
            if 'num_batches_tracked' in k:
                continue
            else:
                model_key[k] = v

        model_value = {}
        key_index = 0
        for k,v in pretrain_model.items():
            if 'features' not in k:
                break
            model_value[key_index] = v
            key_index+=1

        key_i = 0
        for k,v in model_key.items():
            if key_i > len(model_value)-1:
                break
            model_key[k] = model_value[key_i]
            key_i+=1

        state_dict = {k: v for k, v in model_key.items() if k in model.keys()}


        model.update(state_dict)

        self.load_state_dict(model)

def get_msfcnetvgg(num_layers, heads, head_conv):

    model = VGGNet(heads,head_conv,num_layers)
    model.init_weight()
    return model


