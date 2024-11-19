import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, pad, dilation):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(planes),
        )
        if planes != inplanes or stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=stride, stride=stride, padding=pad, dilation=dilation, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        x = self.downsample(x)
        out += x
        return out


class FusionBlock(nn.Module):
    def __init__(self, inplanes_list=[256, 128], planes=128, upsample=True):
        super(FusionBlock, self).__init__()
        self.out_conv = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=True)
        self.in_conv = nn.ModuleList()
        for channels in inplanes_list:
            self.in_conv.append(ResBlock(channels, planes, stride=1, pad=1, dilation=1))
        self.upsample = upsample

    def forward(self, *xs):
        output = xs[0]
        output = self.in_conv[0](output)

        if len(xs) == 2:
            # 存在短连接通路, 先进行预处理
            res = xs[1]
            res = self.in_conv[1](res)
            output += res

        if self.upsample:
            output = nn.functional.interpolate(
                output, scale_factor=2, mode="bilinear", align_corners=True,
            )

        output = self.out_conv(output)

        return output
