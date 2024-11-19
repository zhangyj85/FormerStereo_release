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
from Models.FormerStereo.vit_backbones.vit import _make_pretrained_dinov2_vits14_518, forward_vit


class DINOv2_multi_layers(nn.Module):
    """official PSMNet with cosine similarity"""
    def __init__(self, config):
        super(DINOv2_multi_layers, self).__init__()
        self.maxdisp = config['model']['max_disp']

        self.feature_extraction = _make_pretrained_dinov2_vits14_518(
            hooks=[1,3,5,7],
            use_readout="ignore",
            enable_attention_hooks=False
        )
        for p in self.feature_extraction.model.parameters():
            p.requires_grad = False
        self.feature_extraction.model.eval()

    def forward(self, left, right):

        _, _, H, W = left.shape

        x_layer_1, x_layer_2, x_layer_3, x_layer_4 = forward_vit(self.feature_extraction, left)
        y_layer_1, y_layer_2, y_layer_3, y_layer_4 = forward_vit(self.feature_extraction, right)

        x_layer_1 = F.upsample(x_layer_1, [H // 4, W // 4], mode="bilinear")
        x_layer_2 = F.upsample(x_layer_2, [H // 4, W // 4], mode="bilinear")
        x_layer_3 = F.upsample(x_layer_3, [H // 4, W // 4], mode="bilinear")
        x_layer_4 = F.upsample(x_layer_4, [H // 4, W // 4], mode="bilinear")
        y_layer_1 = F.upsample(y_layer_1, [H // 4, W // 4], mode="bilinear")
        y_layer_2 = F.upsample(y_layer_2, [H // 4, W // 4], mode="bilinear")
        y_layer_3 = F.upsample(y_layer_3, [H // 4, W // 4], mode="bilinear")
        y_layer_4 = F.upsample(y_layer_4, [H // 4, W // 4], mode="bilinear")

        cost_4 = build_gwc_volume(x_layer_4, y_layer_4, self.maxdisp // 4, num_groups=1, norm="cosine")
        cost_3 = build_gwc_volume(x_layer_3, y_layer_3, self.maxdisp // 4, num_groups=1, norm="cosine")
        cost_2 = build_gwc_volume(x_layer_2, y_layer_2, self.maxdisp // 4, num_groups=1, norm="cosine")
        cost_1 = build_gwc_volume(x_layer_1, y_layer_1, self.maxdisp // 4, num_groups=1, norm="cosine")
        cost = cost_1 * cost_2 * cost_3 * cost_4
        cost = F.upsample(cost, [self.maxdisp, H, W], mode="trilinear")
        pred = torch.argmax(cost, dim=2)
        # uncertainty = torch.max(cost, dim=2)[0] < 0.85**4
        pred[pred < 1] = 1
        # pred[uncertainty] = 0
        # pred_up = F.upsample(pred.float() * 4, [H, W], mode="nearest")

        return {"disparity": pred.float()}


class DINOv2_single_layer(nn.Module):
    """official PSMNet with cosine similarity"""
    def __init__(self, config):
        super(DINOv2_single_layer, self).__init__()
        self.maxdisp = config['model']['max_disp']

        self.feature_extraction = _make_pretrained_dinov2_vits14_518(
            hooks=[1,3,5,7],
            use_readout="ignore",
            enable_attention_hooks=False
        )
        for p in self.feature_extraction.model.parameters():
            p.requires_grad = False
        self.feature_extraction.model.eval()

    def forward(self, left, right):

        _, _, H, W = left.shape

        x_layer_1, x_layer_2, x_layer_3, x_layer_4 = forward_vit(self.feature_extraction, left)
        y_layer_1, y_layer_2, y_layer_3, y_layer_4 = forward_vit(self.feature_extraction, right)

        x_layer_4 = F.upsample(x_layer_4, [H // 4, W // 4], mode="bilinear")
        y_layer_4 = F.upsample(y_layer_4, [H // 4, W // 4], mode="bilinear")

        cost_4 = build_gwc_volume(x_layer_4, y_layer_4, self.maxdisp // 4, num_groups=1, norm="cosine")
        cost = cost_4
        cost = F.upsample(cost, [self.maxdisp, H, W], mode="trilinear")
        pred = torch.argmax(cost, dim=2)
        # uncertainty = torch.max(cost, dim=2)[0] < 0.85**4
        pred[pred < 1] = 1
        # pred[uncertainty] = 0
        # pred_up = F.upsample(pred.float() * 4, [H, W], mode="nearest")

        return {"disparity": pred.float()}


class PSMNet_feats(nn.Module):
    def __init__(self, config):
        super(PSMNet_feats, self).__init__()
        self.maxdisp = config['model']['max_disp']

        self.feature_extraction = feature_extraction()

        self.weight_load()

    def weight_load(self):
        # path = "Models/PSMNet/pretrained_model_KITTI2015.tar"
        path = "Models/PSMNet/pretrained_sceneflow_new.tar"
        # states['state_dict'] 里面记录了参数, 但是要把 module 去掉才能 match
        states = torch.load(path, map_location=lambda storage, loc: storage)
        weights = states['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in weights.items():
            name = k[7:] if 'module' in k else k
            new_state_dict[name] = v

        self.load_state_dict(new_state_dict, strict=False)

    def forward(self, left, right):

        _, _, H, W = left.shape
        refimg_fea     = self.feature_extraction(left)
        targetimg_fea  = self.feature_extraction(right)

        #matching
        cost = build_gwc_volume(refimg_fea, targetimg_fea, self.maxdisp//4, 1, "cosine")
        cost = F.upsample(cost, [self.maxdisp, H, W], mode="trilinear")
        pred = torch.argmax(cost, dim=2)
        # uncertainty = torch.max(cost, dim=2)[0] < 0.85**4
        pred[pred < 1] = 1

        return {"disparity": pred.float()}

