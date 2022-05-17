import torch
import torch.nn as nn
import torch.nn.functional as F
from deform_conv.modules.deform_conv import DeformConvPack


class GatedConv2dWithActivation(nn.Module):
    """
    Gated Convlution layer with activation
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True, activation=nn.ReLU):
        super(GatedConv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):
        # return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x


class UpConvWithActivation(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale, mode='bilinear', batch_norm=True, activation=nn.ReLU):
        super().__init__()
        self.scale = scale
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=1, padding=1, bias=False)
        self.batch_norm = batch_norm
        self.batchnorm2d = nn.BatchNorm2d(out_channels)
        self.activation = activation()

    def forward(self, x):
        if self.mode == 'bilinear':
            x = F.interpolate(x, scale_factor=self.scale,
                              mode=self.mode, align_corners=True)
        else:
            x = F.interpolate(x, scale_factor=self.scale,
                              mode=self.mode)
        x = self.conv(x)
        x = self.activation(x)
        if self.batch_norm:
            x = self.batchnorm2d(x)

        return x


class ConvWithActivation(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, batch_norm=True, activation=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=1, padding=1, bias=False)
        self.batch_norm = batch_norm
        self.batchnorm2d = nn.BatchNorm2d(out_channels)
        self.activation = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        if self.batch_norm:
            x = self.batchnorm2d(x)

        return x


class DeformableConvWithActivation(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=True, activation=nn.ReLU):
        super().__init__()
        self.conv = DeformConvPack(in_channels, out_channels, kernel_size, stride=stride,
                                   padding=1, bias=False)
        self.batch_norm = batch_norm
        self.batchnorm2d = nn.BatchNorm2d(out_channels)
        self.activation = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        if self.batch_norm:
            x = self.batchnorm2d(x)

        return x


class GatedDeformConvWithActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True, activation=nn.ReLU):
        super().__init__()
        self.batch_norm = batch_norm
        self.activation = activation()
        self.conv2d = DeformConvPack(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):
        # return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x
