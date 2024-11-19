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
from Models.FormerStereo.vit_backbones.models import ViT_DenseDPT_v4_2 as ViT_Dense
from Models.FormerStereo.loss import stereo_infoNCE


class Former(nn.Module):
    """official PSMNet with cosine similarity"""
    def __init__(self, config):
        super(Former, self).__init__()
        self.maxdisp = config['model']['max_disp']

        self.feature_extraction = ViT_Dense(
            backbone="dinov2_vitl14_518",
            features=[128]*4,
            out_feats=32,
        )

        for m in self.modules():
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

        self.feature_extraction.weight_load()

    def forward(self, left, right):

        _, _, H, W = left.shape

        refimg_fea, targetimg_fea, recon_left, recon_right = self.feature_extraction(left, right)

        output = {}
        if self.training:
            output['init_cost_volume'] = build_gwc_volume(refimg_fea, targetimg_fea, W // 4, num_groups=1, norm="cosine")
            output['recon_loss'] = 1.0 * (F.mse_loss(left, recon_left) + F.mse_loss(right, recon_right))

        cost = build_gwc_volume(refimg_fea, targetimg_fea, self.maxdisp // 4, num_groups=1, norm="cosine").detach()
        cost = F.upsample(cost, [self.maxdisp, H, W], mode="trilinear")
        pred = torch.argmax(cost, dim=2)
        pred[pred < 1] = 1
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
        disp_true = data_batch["gt1"].to('cuda', non_blocking=True)
        mask = (self.min_disp < disp_true) & (disp_true < self.max_disp)
        mask.detach_()

        loss  = stereo_infoNCE(training_output['init_cost_volume'], disp_true, mask)
        loss += training_output['recon_loss']

        return loss
