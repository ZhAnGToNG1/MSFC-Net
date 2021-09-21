import torch
import torch.nn as nn
from models.n_utils.channel_shuffle import channel_shuffle


class FeatureAlign(nn.Module):
    def __init__(self,inchannels):
        super(FeatureAlign, self).__init__()
        self.inchannels = inchannels

        self.conv_deep  = nn.Conv2d(self.inchannels,self.inchannels,kernel_size=3,stride=1,padding=1)
        self.depthwise_conv = nn.Conv2d(self.inchannels * 2, self.inchannels, kernel_size=3, stride=1,
                                        groups = 2, bias=False, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.bn_h = nn.BatchNorm2d(self.inchannels)
        self.bn = nn.BatchNorm2d(self.inchannels)
        self.relu = nn.ReLU(inplace=True)



    def forward(self, shallow_feature, deep_feature):

        attention = self.sigmoid(self.conv_deep(deep_feature))
        shallow_infor = torch.mul(attention,shallow_feature)
        shallow_infor = self.bn_h(shallow_infor)
        fusion_fea = torch.cat([deep_feature,shallow_infor],dim=1)  # 256 channel --->512 channel
        fusion_fea = channel_shuffle(fusion_fea, groups=2)
        fusion_fea = self.depthwise_conv(fusion_fea)
        fusion_fea = self.bn(fusion_fea)
        fusion_fea = self.relu(fusion_fea)
        return fusion_fea




