import torch

def channel_shuffle(x,groups):
    bs , num_channels, h ,w  = x.data.size()
    channels_per_groups = num_channels // groups
    x = x.view(bs,groups,channels_per_groups ,h ,w)
    x = torch.transpose(x,1,2).contiguous()
    x = x.view(bs,-1,h,w)

    return x
