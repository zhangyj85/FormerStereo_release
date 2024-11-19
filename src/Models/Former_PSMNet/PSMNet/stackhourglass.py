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
from .submodule import *
from Models.FormerStereo.vit_backbones import ViT_Dense
from Models.FormerStereo.submodule import build_gwc_volume, build_concat_volume


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes*2)) #+conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes)) #+x

    def forward(self, x ,presqu, postsqu):
        
        out  = self.conv1(x) #in:1/4 out:1/8
        pre  = self.conv2(out) #in:1/8 out:1/8
        if postsqu is not None:
           pre = F.relu(pre + postsqu, inplace=True)
        else:
           pre = F.relu(pre, inplace=True)

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16

        if presqu is not None:
           post = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8
        else:
           post = F.relu(self.conv5(out)+pre, inplace=True) 

        out  = self.conv6(post)  #in:1/8 out:1/4

        return out, pre, post


class PSMNet_v3_reconstruction(nn.Module):
    """
    use normalized concat volume
    with cosine constraint
    """
    def __init__(self, config):
        super(PSMNet_v3_reconstruction, self).__init__()
        self.maxdisp = config['model']['max_disp']

        self.feature_extraction = ViT_Dense(
            backbone="dinov2_vitl14_518",
            # backbone="sam_vitl16_1024",
            features=[128] * 4,
            out_feats=32,
            readout="ignore",
            enable_attention_hooks=False
        )

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.feature_extraction.weight_load()

    def forward(self, left, right):

        refimg_fea, targetimg_fea, recon_left, recon_right = self.feature_extraction(left, right)
        refimg_fea = F.normalize(refimg_fea, dim=1)
        targetimg_fea = F.normalize(targetimg_fea, dim=1)

        cost = build_concat_volume(refimg_fea, targetimg_fea, maxdisp=self.maxdisp // 4)

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1+cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2+cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3+cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if self.training:
            cost1 = F.upsample(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
            cost2 = F.upsample(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')

            cost1 = torch.squeeze(cost1,1)
            pred1 = F.softmax(cost1,dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)

            cost2 = torch.squeeze(cost2,1)
            pred2 = F.softmax(cost2,dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)

            # corr_cost = build_gwc_volume(refimg_fea, targetimg_fea, maxdisp=self.maxdisp//4, num_groups=1, norm="cosine")
            # corr_cost = build_gwc_volume(refimg_fea, targetimg_fea, maxdisp=refimg_fea.shape[-1], num_groups=1, norm="cosine")

        cost3 = F.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3,1)
        pred3 = F.softmax(cost3,dim=1)
        #For your information: This formulation 'softmax(c)' learned "similarity"
        #while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        #However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = disparityregression(self.maxdisp)(pred3)

        output = {}
        output['disparity'] = pred3
        if self.training:
            output['training_output'] = [pred3, pred2, pred1]
            output['init_cost_volume'] = build_gwc_volume(refimg_fea, targetimg_fea, left.shape[-1]//4, 1, "cosine") # cost
            if recon_left is not None:
                output['recon_loss'] = 1.0 * (F.mse_loss(left, recon_left) + F.mse_loss(right, recon_right))

        return output


class PSMNet_v3_reconstruction_frozen(nn.Module):
    """
    use normalized concat volume
    with cosine constraint
    """
    def __init__(self, config):
        super(PSMNet_v3_reconstruction_frozen, self).__init__()
        self.maxdisp = config['model']['max_disp']

        self.feature_extraction = ViT_Dense(
            backbone="dinov2_vitl14_518",
            features=[128] * 4,
            out_feats=32,
            readout="ignore",
            enable_attention_hooks=False
        )

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.feature_extraction.weight_load()
        for p in self.feature_extraction.parameters():
            p.requires_grad = False

    def forward(self, left, right):

        self.feature_extraction.eval()
        with torch.no_grad():
            refimg_fea, targetimg_fea, recon_left, recon_right = self.feature_extraction(left, right)
            refimg_fea = F.normalize(refimg_fea, dim=1)
            targetimg_fea = F.normalize(targetimg_fea, dim=1)

        cost = build_concat_volume(refimg_fea, targetimg_fea, maxdisp=self.maxdisp // 4)

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1+cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2+cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3+cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if self.training:
            cost1 = F.upsample(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
            cost2 = F.upsample(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')

            cost1 = torch.squeeze(cost1,1)
            pred1 = F.softmax(cost1,dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)

            cost2 = torch.squeeze(cost2,1)
            pred2 = F.softmax(cost2,dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)

            # corr_cost = build_gwc_volume(refimg_fea, targetimg_fea, maxdisp=self.maxdisp//4, num_groups=1, norm="cosine")
            # corr_cost = build_gwc_volume(refimg_fea, targetimg_fea, maxdisp=refimg_fea.shape[-1], num_groups=1, norm="cosine")

        cost3 = F.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3,1)
        pred3 = F.softmax(cost3,dim=1)
        #For your information: This formulation 'softmax(c)' learned "similarity"
        #while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        #However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = disparityregression(self.maxdisp)(pred3)

        output = {}
        output['disparity'] = pred3
        if self.training:
            output['training_output'] = [pred3, pred2, pred1]
            # output['init_cost_volume'] = cost
            # # 引入L_cos后, 应避免使用非对称的遮挡增强, 否则会导致该 loss 学习不到有效的关联特征
            # output['recon_loss'] = F.mse_loss(left, recon_left) + F.mse_loss(right, recon_right)

        return output
