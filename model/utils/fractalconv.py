import torch.nn as nn


def conv3x3(in_channels, out_channels, **kwargs):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, **kwargs)
    layer = init_conv_weights(layer)
    return layer


def init_conv_weights(layer, weights_std=0.01, bias=0):
    nn.init.normal_(layer.weight, std=weights_std)
    nn.init.constant_(layer.bias, val=bias)

    return layer
