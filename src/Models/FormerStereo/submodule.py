from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.autograd.function import Function
import torch.nn.functional as F
import numpy as np


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def groupwise_correlation_norm(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    fea1 = fea1.view([B, num_groups, channels_per_group, H, W])
    fea2 = fea2.view([B, num_groups, channels_per_group, H, W])
    cost = ((fea1/(torch.norm(fea1, 2, 2, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 2, True)+1e-05))).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def groupwise_correlation_cosine(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    fea1 = fea1.view([B, num_groups, channels_per_group, H, W])
    fea2 = fea2.view([B, num_groups, channels_per_group, H, W])
    cost = F.cosine_similarity(fea1, fea2, dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def groupwise_correlation_cosine2(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    fea1 = fea1.view([B, num_groups, channels_per_group, H, W])
    fea2 = fea2.view([B, num_groups, channels_per_group, H, W])
    cost = F.cosine_similarity(fea1, fea2, dim=2)
    assert cost.shape == (B, num_groups, H, W)
    """normalize to (0,1)"""
    return (cost + 1.) / 2.


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups, norm=None):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    if norm == "cosine":
        fn = groupwise_correlation_cosine
    elif norm == "l2":
        fn = groupwise_correlation_norm
    elif norm == "cosine2":
        fn = groupwise_correlation_cosine2
    elif norm == "none":
        fn = groupwise_correlation
    else:
        raise NotImplementedError

    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = fn(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i], num_groups)
        else:
            volume[:, :, i, :, :] = fn(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class learnable_upsample(nn.Module):
    """Upsample features according to its features"""
    def __init__(self, in_chans, out_chans, rate=4):
        super().__init__()
        # map the input into the output channels
        self.mlp = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 1, stride=1, padding=0, bias=False),     # 通道压缩
            nn.GroupNorm(num_groups=1, num_channels=out_chans), nn.GELU(),                    # 增加非线性
            nn.Conv2d(out_chans, out_chans, 1, stride=1, padding=0),
        )
        # upsample flow
        self.mask = nn.Sequential(
            nn.Conv2d(in_chans, 256, 1, stride=1, padding=0, bias=False),  # 通道压缩
            nn.GroupNorm(num_groups=1, num_channels=256), nn.GELU(),  # 增加非线性
            nn.Conv2d(256, (rate ** 2) * 9, 3, stride=1, padding=1, bias=False),
        )
        self.rate = rate

    def forward(self, x):
        rate = self.rate
        flow = self.mlp(x)
        mask = self.mask(x)

        N, C, H, W = flow.shape
        mask = mask.view(N, 1, 9, rate, rate, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(flow, [3, 3], padding=1)  # 注意, 由于不是光流/视差, 对上采样的结果不需要乘以倍率
        up_flow = up_flow.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_flow = up_flow.reshape(N, C, rate * H, rate * W)
        return up_flow.contiguous()
