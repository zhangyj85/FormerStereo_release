from __future__ import print_function

import os.path
import time
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import cv2
from Models.FormerStereo.submodule import build_gwc_volume, build_concat_volume
# from Models.FormerStereo.vit_backbones.models import ViT_DenseDPT_v4_2 as ViT_Dense
from Models.FormerStereo.loss import stereo_infoNCE
from Models.FormerStereo.vit_backbones.vit import _make_pretrained_dinov2_vitl14_518, forward_vit


class Former(nn.Module):
    """official PSMNet with cosine similarity"""
    def __init__(self, config):
        super(Former, self).__init__()
        self.maxdisp = config['model']['max_disp']

        self.feature_extraction = _make_pretrained_dinov2_vitl14_518(
            hooks=[5, 11, 17, 23],
            use_readout="ignore",
            enable_attention_hooks=False
        )
        for p in self.feature_extraction.model.parameters():
            p.requires_grad = False
        self.feature_extraction.model.eval()

        vit_features = self.feature_extraction.model.vit_features
        self.upsample1 = nn.ConvTranspose2d(in_channels=vit_features, out_channels=256, kernel_size=4, stride=4, padding=0, bias=True, dilation=1, groups=1)
        self.upsample2 = nn.ConvTranspose2d(in_channels=vit_features, out_channels=256, kernel_size=4, stride=4, padding=0, bias=True, dilation=1, groups=1)
        self.upsample3 = nn.ConvTranspose2d(in_channels=vit_features, out_channels=256, kernel_size=4, stride=4, padding=0, bias=True, dilation=1, groups=1)
        self.upsample4 = nn.ConvTranspose2d(in_channels=vit_features, out_channels=256, kernel_size=4, stride=4, padding=0, bias=True, dilation=1, groups=1)

        for block in [self.upsample1, self.upsample2, self.upsample3, self.upsample4]:
            m = block.modules()
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, left, right):

        _, _, H, W = left.shape
        self.feature_extraction.model.eval()

        x_layer_1, x_layer_2, x_layer_3, x_layer_4 = forward_vit(self.feature_extraction, left)
        y_layer_1, y_layer_2, y_layer_3, y_layer_4 = forward_vit(self.feature_extraction, right)

        x_layer_1, y_layer_1 = self.upsample1(x_layer_1), self.upsample1(y_layer_1)
        x_layer_2, y_layer_2 = self.upsample2(x_layer_2), self.upsample2(y_layer_2)
        x_layer_3, y_layer_3 = self.upsample3(x_layer_3), self.upsample3(y_layer_3)
        x_layer_4, y_layer_4 = self.upsample4(x_layer_4), self.upsample4(y_layer_4)

        x_layer_1 = F.upsample(x_layer_1, [H // 4, W // 4], mode="bilinear")
        x_layer_2 = F.upsample(x_layer_2, [H // 4, W // 4], mode="bilinear")
        x_layer_3 = F.upsample(x_layer_3, [H // 4, W // 4], mode="bilinear")
        x_layer_4 = F.upsample(x_layer_4, [H // 4, W // 4], mode="bilinear")
        y_layer_1 = F.upsample(y_layer_1, [H // 4, W // 4], mode="bilinear")
        y_layer_2 = F.upsample(y_layer_2, [H // 4, W // 4], mode="bilinear")
        y_layer_3 = F.upsample(y_layer_3, [H // 4, W // 4], mode="bilinear")
        y_layer_4 = F.upsample(y_layer_4, [H // 4, W // 4], mode="bilinear")

        refimg_fea = torch.cat([x_layer_1, x_layer_2, x_layer_3, x_layer_4], dim=1)
        targetimg_fea = torch.cat([y_layer_1, y_layer_2, y_layer_3, y_layer_4], dim=1)

        output = {}
        output['init_cost_volume'] = build_gwc_volume(refimg_fea, targetimg_fea, self.maxdisp // 4, num_groups=1, norm="cosine")

        cost = output['init_cost_volume'].detach()
        cost = F.upsample(cost, [self.maxdisp, H, W], mode="trilinear")
        pred = torch.argmax(cost, dim=2)
        pred[pred < 1e-3] = 1e-3
        output['disparity'] = pred.float()

        return output


class loss_func(nn.Module):

    def __init__(self, config):
        super(loss_func, self).__init__()
        self.max_disp = config['model']['max_disp']
        self.min_disp = config['model']['min_disp']

    def forward(self, data_batch, training_output):
        # target: B1HW, (min_disp, max_disp)
        # output: a dict containing multi-scale disp map
        disp_true = data_batch["gt1"]
        mask = (self.min_disp < disp_true) & (disp_true < self.max_disp)
        mask.detach_()

        loss  = stereo_infoNCE(training_output['init_cost_volume'], disp_true, mask)

        return loss
