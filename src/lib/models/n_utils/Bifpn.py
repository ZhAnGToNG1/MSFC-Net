import torch.nn as nn
import torch
import torch.nn.functional as F
from models.n_utils.separableconvblock import SeparableConvBlock,MaxPool2dStaticSamePadding,Swish

class BiFPN(nn.Module):

    def __init__(self, num_channels, epsilon = 1e-4, attention = True):

        super(BiFPN, self).__init__()
        self.epsilon = epsilon

        #Conv layers

        self.conv6_up = SeparableConvBlock(num_channels)
        self.conv5_up = SeparableConvBlock(num_channels)
        self.conv4_up = SeparableConvBlock(num_channels)
        self.conv3_up = SeparableConvBlock(num_channels)
        self.conv4_down = SeparableConvBlock(num_channels)
        self.conv5_down = SeparableConvBlock(num_channels)
        self.conv6_down = SeparableConvBlock(num_channels)
        self.conv7_down = SeparableConvBlock(num_channels)


        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)


        self.swish = Swish()

        # weight
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype= torch.float32), requires_grad= True)
        self.p6_w1_relu = nn.ReLU()

        self.p5_w1 = nn.Parameter(torch.ones(2, dtype= torch.float32), requires_grad= True)
        self.p5_w1_relu = nn.ReLU()

        self.p4_w1 = nn.Parameter(torch.ones(2, dtype= torch.float32), requires_grad= True)
        self.p4_w1_relu = nn.ReLU()

        self.p3_w1 = nn.Parameter(torch.ones(2, dtype= torch.float32), requires_grad= True)
        self.p3_w1_relu = nn.ReLU()



        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()

        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()

        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()

        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()


        self.attention = attention






    def forward(self, inputs):

        p3_in, p4_in, p5_in, p6_in, p7_in = inputs


        # weight for p6_0 and p7_0 to p6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)

        p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * F.interpolate(p7_in,scale_factor=2 ,mode='bilinear',align_corners=True)))

        # weight for p5_0 and p6_1 to p5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)

        p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * F.interpolate(p6_up,scale_factor=2 ,mode='bilinear',align_corners=True)))

        #weight for p4_0 and p5_1 to p4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)

        p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * F.interpolate(p5_up,scale_factor=2 ,mode='bilinear',align_corners=True)))

        # weight for p3_0 and p4_1 to p3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)


        p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * F.interpolate(p4_up,scale_factor=2 ,mode='bilinear',align_corners=True)))



        #weights for p4_0 p4_1 and p3_2 to p4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim= 0) + self.epsilon)

        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))

        #weights for p5_0 p5_1 and p4_2 to p5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)

        weight = p5_w2 / (torch.sum(p5_w2 ,dim = 0) + self.epsilon)

        p5_out = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out)))

        #weights for p6_0 p6_1 and p5_2 for p6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)

        weight = p6_w2 / (torch.sum(p6_w2,dim = 0) + self.epsilon)

        p6_out = self.conv6_down(
            self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out)))

        # weight for p7_0 and p6_2 to p7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)

        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)

        p7_out = self.conv7_down(self.swish(weight[0]*p7_in + weight[1] * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out














