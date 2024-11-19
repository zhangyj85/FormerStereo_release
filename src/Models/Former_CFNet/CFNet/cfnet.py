from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from .submodule import *
import math
from Models.FormerStereo.submodule import build_gwc_volume as build_gwc_norm_volume


class Former(nn.Module):
    def __init__(self, maxdisp, concat_feature_channel=16):
        super().__init__()
        self.maxdisp = max(256, maxdisp)  # 最小的视差范围为256
        self.concat_channels = concat_feature_channel
        # DINOv2 及 Former
        from Models.FormerStereo.vit_backbones.models import ViT_DenseDPT_v4_2_CFNet2
        self.feature_extraction = ViT_DenseDPT_v4_2_CFNet2(
            backbone="dinov2_vitb14_518",
            # backbone="dam_vitl14_518",
            features=[128] * 4,
            out_feats=concat_feature_channel,
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
            # elif isinstance(m, nn.Linear):
            #     m.bias.data.zero_()
        self.feature_extraction.weight_load()    # 加载 DINOv2 参数

    def forward(self, left, right):
        _, _, H, W = left.shape
        features_left, features_right = self.feature_extraction(left, right)

        outputs = {}
        if self.training:
            outputs['training_output'] = []
            outputs['recon_loss'] = F.mse_loss(left, features_left['recon_img']) + F.mse_loss(right, features_right['recon_img'])
            outputs['init_cost_volume'] = [
                build_gwc_norm_volume(features_left['feats_cat'], features_right['feats_cat'], W// 4, 1, "cosine"),
            ]

        cost = build_gwc_norm_volume(features_left['feats_cat'], features_right['feats_cat'], self.maxdisp // 4, num_groups=1, norm="cosine").detach()
        cost = F.upsample(cost, [self.maxdisp, H, W], mode="trilinear")
        pred = torch.argmax(cost, dim=2)
        pred[pred < 1] = 1
        outputs['disparity'] = pred.float()

        return outputs


class feature_extraction(nn.Module):
    def __init__(self, concat_feature_channel=16):
        super(feature_extraction, self).__init__()

        self.former = Former(maxdisp=256, concat_feature_channel=concat_feature_channel)

        # 1/2 分支的额外网络
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       Mish(),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       Mish(),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       Mish())
        self.layer2 = self._make_layer(BasicBlock, 64, 1, 1, 1, 1)
        self.upconv3 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(128, 64, 3, 1, 1, 1),
                                     Mish())
        self.iconv2 = nn.Sequential(convbn(128, 64, 3, 1, 1, 1),
                                    Mish())

        # 构造cost volume的分支输出
        self.gw2 = nn.Sequential(convbn(64, 80, 3, 1, 1, 1),
                                          Mish(),
                                          nn.Conv2d(80, 80, kernel_size=1, padding=0, stride=1, bias=False))
        self.gw3 = nn.Sequential(convbn(128, 160, 3, 1, 1, 1),
                                 Mish(),
                                 nn.Conv2d(160, 160, kernel_size=1, padding=0, stride=1, bias=False))
        self.gw4 = nn.Sequential(convbn(128, 160, 3, 1, 1, 1),
                                 Mish(),
                                 nn.Conv2d(160, 160, kernel_size=1, padding=0, stride=1, bias=False))
        self.gw5 = nn.Sequential(convbn(128, 320, 3, 1, 1, 1),
                                 Mish(),
                                 nn.Conv2d(320, 320, kernel_size=1, padding=0, stride=1, bias=False))
        self.gw6 = nn.Sequential(convbn(128, 320, 3, 1, 1, 1),
                                 Mish(),
                                 nn.Conv2d(320, 320, kernel_size=1, padding=0, stride=1, bias=False))

        self.concat2 = nn.Sequential(convbn(64, 32, 3, 1, 1, 1),
                                      Mish(),
                                      nn.Conv2d(32, concat_feature_channel // 2, kernel_size=1, padding=0, stride=1, bias=False))
        # self.concat3 = nn.Sequential(convbn(128, 128, 3, 1, 1, 1),
        #                               Mish(),
        #                               nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1, bias=False))
        self.concat4 = nn.Sequential(convbn(128, 128, 3, 1, 1, 1),
                                      Mish(),
                                      nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1, bias=False))
        self.concat5 = nn.Sequential(convbn(128, 128, 3, 1, 1, 1),
                                     Mish(),
                                     nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1, bias=False))
        self.concat6 = nn.Sequential(convbn(128, 128, 3, 1, 1, 1),
                                     Mish(),
                                     nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1, bias=False))

    def load_former(self, path="/media/zhangyj85/Data/CFNet/Former/FormerStereo/iter_33210.pth", frozen=True):
        state_dict = torch.load(path, map_location="cpu")
        state_dict = state_dict['model_state']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.backbone.', '')
            new_state_dict[name] = v
        self.former.load_state_dict(new_state_dict, strict=True)

        if frozen:
            for p in self.former.parameters():
                p.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x, y):
        # 得到 1/2 的特征图
        x_l2 = self.firstconv(x)
        x_l2 = self.layer2(x_l2)    # 1/2
        y_l2 = self.firstconv(y)
        y_l2 = self.layer2(y_l2)    # 1/2

        # 得到多尺度融合特征, 1/4, 1/8, 1/16, 1/32
        x_outputs, y_outputs = self.former.feature_extraction(x, y)
        x_l3, x_l4, x_l5, x_l6 = x_outputs['layer_1'], x_outputs['layer_2'], x_outputs['layer_3'], x_outputs['layer_4']
        y_l3, y_l4, y_l5, y_l6 = y_outputs['layer_1'], y_outputs['layer_2'], y_outputs['layer_3'], y_outputs['layer_4']

        # 对 1/2 特征进行融合
        x_l2 = torch.cat((x_l2, self.upconv3(x_l3)), dim=1)
        x_l2 = self.iconv2(x_l2)
        y_l2 = torch.cat((y_l2, self.upconv3(y_l3)), dim=1)
        y_l2 = self.iconv2(y_l2)

        # 转化获取所需特征
        x_gw2 = self.gw2(x_l2)
        x_gw3 = self.gw3(x_l3)
        x_gw4 = self.gw4(x_l4)
        x_gw5 = self.gw5(x_l5)
        x_gw6 = self.gw6(x_l6)

        y_gw2 = self.gw2(y_l2)
        y_gw3 = self.gw3(y_l3)
        y_gw4 = self.gw4(y_l4)
        y_gw5 = self.gw5(y_l5)
        y_gw6 = self.gw6(y_l6)

        x_cat2 = self.concat2(x_l2)
        # x_cat3 = self.concat3(x_l3)
        x_cat3 = x_outputs['feats_cat']
        x_cat4 = self.concat4(x_l4)
        x_cat5 = self.concat5(x_l5)
        x_cat6 = self.concat6(x_l6)

        y_cat2 = self.concat2(y_l2)
        # y_cat3 = self.concat3(y_l3)
        y_cat3 = y_outputs['feats_cat']
        y_cat4 = self.concat4(y_l4)
        y_cat5 = self.concat5(y_l5)
        y_cat6 = self.concat6(y_l6)

        x_return = {"gw2": x_gw2, "gw3": x_gw3, "gw4": x_gw4, "gw5": x_gw5, "gw6": x_gw6,
                    "concat_feature2": x_cat2, "concat_feature3": x_cat3, "concat_feature4": x_cat4,
                    "concat_feature5": x_cat5, "concat_feature6": x_cat6,
                    "recon_img": x_outputs['recon_img']}
        y_return = {"gw2": y_gw2, "gw3": y_gw3, "gw4": y_gw4, "gw5": y_gw5, "gw6": y_gw6,
                    "concat_feature2": y_cat2, "concat_feature3": y_cat3, "concat_feature4": y_cat4,
                    "concat_feature5": y_cat5, "concat_feature6": y_cat6,
                    "recon_img": y_outputs['recon_img']}
        return x_return, y_return


class hourglassup(nn.Module):
    def __init__(self, in_channels):
        super(hourglassup, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels * 2, kernel_size=3, stride=2,
                                   padding=1, bias=False)

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   Mish())

        self.conv3 = nn.Conv3d(in_channels * 2, in_channels * 4, kernel_size=3, stride=2,
                               padding=1, bias=False)

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   Mish())

        self.conv8 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.combine1 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 2, 3, 1, 1),
                                   Mish())
        self.combine2 = nn.Sequential(convbn_3d(in_channels * 6, in_channels * 4, 3, 1, 1),
                                      Mish())
        # self.combine3 = nn.Sequential(convbn_3d(in_channels * 6, in_channels * 4, 3, 1, 1),
        #                               Mish())

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)
        # self.redir3 = convbn_3d(in_channels * 4, in_channels * 4, kernel_size=1, stride=1, pad=0)


    def forward(self, x, feature4, feature5):
        conv1 = self.conv1(x)          #1/8
        conv1 = torch.cat((conv1, feature4), dim=1)   #1/8
        conv1 = self.combine1(conv1)   #1/8
        conv2 = self.conv2(conv1)      #1/8

        conv3 = self.conv3(conv2)      #1/16
        conv3 = torch.cat((conv3, feature5), dim=1)   #1/16
        conv3 = self.combine2(conv3)   #1/16
        conv4 = self.conv4(conv3)      #1/16

        conv8 = FMish(self.conv8(conv4) + self.redir2(conv2))
        conv9 = FMish(self.conv9(conv8) + self.redir1(x))

        return conv9


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   Mish())

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   Mish())

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   Mish())

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   Mish())

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        # conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        # conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        conv5 = FMish(self.conv5(conv4) + self.redir2(conv2))
        conv6 = FMish(self.conv6(conv5) + self.redir1(x))

        return conv6


class cfnet(nn.Module):
    def __init__(self, maxdisp, frozen_former=True):
        super(cfnet, self).__init__()
        self.maxdisp = max(256, maxdisp)    # 最小的视差范围为256
        self.use_concat_volume = True
        self.frozen_former = frozen_former
        self.v_scale_s1 = 1
        self.v_scale_s2 = 2
        self.v_scale_s3 = 3
        self.sample_count_s1 = 6
        self.sample_count_s2 = 10
        self.sample_count_s3 = 14
        self.num_groups = 32#40
        self.uniform_sampler = UniformSampler()
        self.spatial_transformer = SpatialTransformer()

        # 定义特征网络
        self.concat_channels = 16#12
        self.feature_extraction = feature_extraction(concat_feature_channel=self.concat_channels)

        # 定义代价聚合网络
        self.dres0 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels*2, 32, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   Mish())

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(32, 32, 3, 1, 1))


        self.dres0_5 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels*2, 64, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(64, 64, 3, 1, 1),
                                   Mish())

        self.dres1_5 = nn.Sequential(convbn_3d(64, 64, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(64, 64, 3, 1, 1))

        self.dres0_6 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels*2, 64, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(64, 64, 3, 1, 1),
                                   Mish())

        self.dres1_6 = nn.Sequential(convbn_3d(64, 64, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(64, 64, 3, 1, 1))

        self.combine1 = hourglassup(32)

        # self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        # self.dres4 = hourglass(32)

        self.confidence0_s3 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels*2 + 1 , 32, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   Mish())

        self.confidence1_s3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.confidence2_s3 = hourglass(32)

        self.confidence3_s3 = hourglass(32)

        self.confidence0_s2 = nn.Sequential(convbn_3d(self.num_groups//2 + self.concat_channels + 1, 16, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(16, 16, 3, 1, 1),
                                   Mish())

        self.confidence1_s2 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(16, 16, 3, 1, 1))

        self.confidence2_s2 = hourglass(16)

        self.confidence3_s2 = hourglass(16)


        # self.confidence0_s1 = nn.Sequential(convbn_3d(self.num_groups // 4 + self.concat_channels // 2 + 1, 16, 3, 1, 1),
        #                                     nn.ReLU(inplace=True),
        #                                     convbn_3d(16, 16, 3, 1, 1),
        #                                     nn.ReLU(inplace=True))
        #
        # self.confidence1_s1 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
        #                                     nn.ReLU(inplace=True),
        #                                     convbn_3d(16, 16, 3, 1, 1))
        #
        # self.confidence2_s1 = hourglass(16)

        # self.confidence3 = hourglass(32)
        #
        # self.confidence4 = hourglass(32)

        self.confidence_classif0_s3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                                    Mish(),
                                                    nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.confidence_classif1_s3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.confidence_classifmid_s3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                                    Mish(),
                                                    nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.confidence_classif0_s2 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                                    Mish(),
                                                    nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))


        self.confidence_classif1_s2 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                                    Mish(),
                                                    nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.confidence_classifmid_s2 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                                    Mish(),
                                                    nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))

        # self.confidence_classif1_s1 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
        #                                             nn.ReLU(inplace=True),
        #                                             nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        # 这些参数都没有梯度, 默认为0
        self.gamma_s3 = 0.  # nn.Parameter(torch.zeros(1))
        self.beta_s3 = 0.   # nn.Parameter(torch.zeros(1))
        self.gamma_s2 = 0.  # nn.Parameter(torch.zeros(1))
        self.beta_s2 = 0.   # nn.Parameter(torch.zeros(1))

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
            # elif isinstance(m, nn.Linear):
                # m.bias.data.zero_()

        if frozen_former:
            self.feature_extraction.load_former(frozen=frozen_former)    # 加载 DINOv2 参数
        else:
            self.feature_extraction.former.feature_extraction.weight_load()

    def generate_search_range(self, sample_count, input_min_disparity, input_max_disparity, scale):
        """
        Description:    Generates the disparity search range.

        Returns:
            :min_disparity: Lower bound of disparity search range
            :max_disparity: Upper bound of disaprity search range.
        """

        min_disparity = torch.clamp(input_min_disparity - torch.clamp((
                sample_count - input_max_disparity + input_min_disparity), min=0) / 2.0, min=0, max=self.maxdisp // (2**scale) - 1)
        max_disparity = torch.clamp(input_max_disparity + torch.clamp(
                sample_count - input_max_disparity + input_min_disparity, min=0) / 2.0, min=0, max=self.maxdisp // (2**scale) - 1)

        return min_disparity, max_disparity

    def generate_disparity_samples(self, min_disparity, max_disparity, sample_count=12):
        """
        Description:    Generates "sample_count" number of disparity samples from the
                                                            search range (min_disparity, max_disparity)
                        Samples are generated by uniform sampler

        Args:
            :min_disparity: LowerBound of the disaprity search range.
            :max_disparity: UpperBound of the disparity search range.
            :sample_count: Number of samples to be generated from the input search range.

        Returns:
            :disparity_samples:
        """
        disparity_samples = self.uniform_sampler(min_disparity, max_disparity, sample_count)

        disparity_samples = torch.cat((torch.floor(min_disparity), disparity_samples, torch.ceil(max_disparity)),
                                      dim=1).long()                   # disparity level = sample_count + 2
        return disparity_samples

    def cost_volume_generator(self, left_input, right_input, disparity_samples, model = 'concat', num_groups = 40):
        """
        Description: Generates cost-volume using left image features, disaprity samples
                                                            and warped right image features.
        Args:
            :left_input: Left Image fetaures
            :right_input: Right Image features
            :disparity_samples: Disaprity samples
            :model : concat or group correlation

        Returns:
            :cost_volume:
            :disaprity_samples:
        """

        right_feature_map, left_feature_map = self.spatial_transformer(left_input,
                                                                       right_input, disparity_samples)
        disparity_samples = disparity_samples.unsqueeze(1).float()
        if model == 'concat':
             cost_volume = torch.cat((left_feature_map, right_feature_map), dim=1)
        else:
             cost_volume = groupwise_correlation_4D(left_feature_map, right_feature_map, num_groups)

        return cost_volume, disparity_samples

    def forward(self, left, right):
        features_left, features_right = self.feature_extraction(left, right)

        gwc_volume4 = build_gwc_volume(features_left["gw4"], features_right["gw4"], self.maxdisp // 8, self.num_groups)
        gwc_volume5 = build_gwc_volume(features_left["gw5"], features_right["gw5"], self.maxdisp // 16, self.num_groups)
        gwc_volume6 = build_gwc_volume(features_left["gw6"], features_right["gw6"], self.maxdisp // 32, self.num_groups)

        if self.use_concat_volume:
            concat_volume4 = build_concat_volume(features_left["concat_feature4"], features_right["concat_feature4"], self.maxdisp // 8)
            concat_volume5 = build_concat_volume(features_left["concat_feature5"], features_right["concat_feature5"], self.maxdisp // 16)
            concat_volume6 = build_concat_volume(features_left["concat_feature6"], features_right["concat_feature6"], self.maxdisp // 32)
            volume4 = torch.cat((gwc_volume4, concat_volume4), 1)
            volume5 = torch.cat((gwc_volume5, concat_volume5), 1)
            volume6 = torch.cat((gwc_volume6, concat_volume6), 1)
        else:
            volume4 = gwc_volume4

        # 构造多尺度融合代价体
        cost0_4 = self.dres0(volume4)                   # 1/8
        cost0_4 = self.dres1(cost0_4) + cost0_4         # 1/8
        cost0_5 = self.dres0_5(volume5)                 # 1/16
        cost0_5 = self.dres1_5(cost0_5) + cost0_5
        cost0_6 = self.dres0_6(volume6)                 # 1/32
        cost0_6 = self.dres1_6(cost0_6) + cost0_6
        # 融合上述代价体
        out1_4 = self.combine1(cost0_4, cost0_5, cost0_6)
        out2_4 = self.dres3(out1_4)
        # 回归视差
        cost2_s4 = self.classif2(out2_4)
        cost2_s4 = torch.squeeze(cost2_s4, 1)
        pred2_possibility_s4 = F.softmax(cost2_s4, dim=1)
        pred2_s4 = disparity_regression(pred2_possibility_s4, self.maxdisp // 8).unsqueeze(1)

        # 计算方差
        pred2_s4_cur = pred2_s4.detach()
        pred2_v_s4 = disparity_variance(pred2_possibility_s4, self.maxdisp // 8, pred2_s4_cur)  # get the variance
        pred2_v_s4 = pred2_v_s4.sqrt()
        mindisparity_s3 = pred2_s4_cur - (self.gamma_s3 + 1) * pred2_v_s4 - self.beta_s3
        maxdisparity_s3 = pred2_s4_cur + (self.gamma_s3 + 1) * pred2_v_s4 + self.beta_s3
        maxdisparity_s3 = F.upsample(maxdisparity_s3 * 2, [left.size()[2] // 4, left.size()[3] // 4], mode='bilinear', align_corners=True)
        mindisparity_s3 = F.upsample(mindisparity_s3 * 2, [left.size()[2] // 4, left.size()[3] // 4], mode='bilinear', align_corners=True)

        # 根据预测结果和方差, 计算下个阶段的视差估计范围 (1/4)
        mindisparity_s3_1, maxdisparity_s3_1 = self.generate_search_range(self.sample_count_s3 + 1, mindisparity_s3, maxdisparity_s3, scale = 2)
        disparity_samples_s3 = self.generate_disparity_samples(mindisparity_s3_1, maxdisparity_s3_1, self.sample_count_s3).float()
        confidence_v_concat_s3, _ = self.cost_volume_generator(features_left["concat_feature3"],
                                                               features_right["concat_feature3"], disparity_samples_s3, 'concat')
        confidence_v_gwc_s3, disparity_samples_s3 = self.cost_volume_generator(features_left["gw3"], features_right["gw3"],
                                                                               disparity_samples_s3, 'gwc', self.num_groups)
        confidence_v_s3 = torch.cat((confidence_v_gwc_s3, confidence_v_concat_s3, disparity_samples_s3), dim=1)

        disparity_samples_s3 = torch.squeeze(disparity_samples_s3, dim=1)

        cost0_s3 = self.confidence0_s3(confidence_v_s3)
        cost0_s3 = self.confidence1_s3(cost0_s3) + cost0_s3

        out1_s3 = self.confidence2_s3(cost0_s3)
        out2_s3 = self.confidence3_s3(out1_s3)

        cost1_s3 = self.confidence_classif1_s3(out2_s3).squeeze(1)
        cost1_s3_possibility = F.softmax(cost1_s3, dim=1)
        pred1_s3 = torch.sum(cost1_s3_possibility * disparity_samples_s3, dim=1, keepdim=True)

        # 根据 1/4 尺度的预测结果, 计算下一轮的方差、范围
        pred1_s3_cur = pred1_s3.detach()
        pred1_v_s3 = disparity_variance_confidence(cost1_s3_possibility, disparity_samples_s3, pred1_s3_cur)
        pred1_v_s3 = pred1_v_s3.sqrt()
        mindisparity_s2 = pred1_s3_cur - (self.gamma_s2 + 1) * pred1_v_s3 - self.beta_s2
        maxdisparity_s2 = pred1_s3_cur + (self.gamma_s2 + 1) * pred1_v_s3 + self.beta_s2
        maxdisparity_s2 = F.upsample(maxdisparity_s2 * 2, [left.size()[2] // 2, left.size()[3] // 2], mode='bilinear', align_corners=True)
        mindisparity_s2 = F.upsample(mindisparity_s2 * 2, [left.size()[2] // 2, left.size()[3] // 2], mode='bilinear', align_corners=True)

        # 根据范围生成 1/2 尺度的代价体
        mindisparity_s2_1, maxdisparity_s2_1 = self.generate_search_range(self.sample_count_s2 + 1, mindisparity_s2, maxdisparity_s2, scale = 1)
        disparity_samples_s2 = self.generate_disparity_samples(mindisparity_s2_1, maxdisparity_s2_1, self.sample_count_s2).float()
        confidence_v_concat_s2, _ = self.cost_volume_generator(features_left["concat_feature2"],
                                                               features_right["concat_feature2"], disparity_samples_s2, 'concat')
        confidence_v_gwc_s2, disparity_samples_s2 = self.cost_volume_generator(features_left["gw2"], features_right["gw2"],
                                                                               disparity_samples_s2, 'gwc', self.num_groups // 2)
        confidence_v_s2 = torch.cat((confidence_v_gwc_s2, confidence_v_concat_s2, disparity_samples_s2), dim=1)

        disparity_samples_s2 = torch.squeeze(disparity_samples_s2, dim=1)

        cost0_s2 = self.confidence0_s2(confidence_v_s2)
        cost0_s2 = self.confidence1_s2(cost0_s2) + cost0_s2

        out1_s2 = self.confidence2_s2(cost0_s2)
        out2_s2 = self.confidence3_s2(out1_s2)

        cost1_s2 = self.confidence_classif1_s2(out2_s2).squeeze(1)
        cost1_s2_possibility = F.softmax(cost1_s2, dim=1)
        pred1_s2 = torch.sum(cost1_s2_possibility * disparity_samples_s2, dim=1, keepdim=True)

        # pred1_v_s2 = disparity_variance_confidence(cost1_s2_possibility, disparity_samples_s2, pred1_s2)
        # pred1_v_s2 = pred1_v_s2.sqrt()

        if self.training:
            cost0_4 = self.classif0(cost0_4)
            cost1_4 = self.classif1(out1_4)

            cost0_4 = F.upsample(cost0_4, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear', align_corners=True)
            cost0_4 = torch.squeeze(cost0_4, 1)
            pred0_4 = F.softmax(cost0_4, dim=1)
            pred0_4 = disparity_regression(pred0_4, self.maxdisp)

            cost1_4 = F.upsample(cost1_4, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear', align_corners=True)
            cost1_4 = torch.squeeze(cost1_4, 1)
            pred1_4 = F.softmax(cost1_4, dim=1)
            pred1_4 = disparity_regression(pred1_4, self.maxdisp)

            pred2_s4 = F.upsample(pred2_s4 * 8, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)
            pred2_s4 = torch.squeeze(pred2_s4, 1)

            cost0_s3 = self.confidence_classif0_s3(cost0_s3).squeeze(1)
            cost0_s3 = F.softmax(cost0_s3, dim=1)
            pred0_s3 = torch.sum(cost0_s3 * disparity_samples_s3, dim=1, keepdim=True)
            pred0_s3 = F.upsample(pred0_s3 * 4, [left.size()[2], left.size()[3]], mode='bilinear',
                                     align_corners=True)
            pred0_s3 = torch.squeeze(pred0_s3, 1)

            costmid_s3 = self.confidence_classifmid_s3(out1_s3).squeeze(1)
            costmid_s3 = F.softmax(costmid_s3, dim=1)
            predmid_s3 = torch.sum(costmid_s3 * disparity_samples_s3, dim=1, keepdim=True)
            predmid_s3 = F.upsample(predmid_s3 * 4, [left.size()[2], left.size()[3]], mode='bilinear',
                                     align_corners=True)
            predmid_s3 = torch.squeeze(predmid_s3, 1)

            pred1_s3_up = F.upsample(pred1_s3 * 4, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)
            pred1_s3_up = torch.squeeze(pred1_s3_up, 1)

            cost0_s2 = self.confidence_classif0_s2(cost0_s2).squeeze(1)
            cost0_s2 = F.softmax(cost0_s2, dim=1)
            pred0_s2 = torch.sum(cost0_s2 * disparity_samples_s2, dim=1, keepdim=True)
            pred0_s2 = F.upsample(pred0_s2 * 2, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)
            pred0_s2 = torch.squeeze(pred0_s2, 1)

            costmid_s2 = self.confidence_classifmid_s2(out1_s2).squeeze(1)
            costmid_s2 = F.softmax(costmid_s2, dim=1)
            predmid_s2 = torch.sum(costmid_s2 * disparity_samples_s2, dim=1, keepdim=True)
            predmid_s2 = F.upsample(predmid_s2 * 2, [left.size()[2], left.size()[3]], mode='bilinear',
                                     align_corners=True)
            predmid_s2 = torch.squeeze(predmid_s2, 1)

            pred1_s2 = F.upsample(pred1_s2 * 2, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)
            pred1_s2 = torch.squeeze(pred1_s2, 1)

            outputs = {}
            outputs['training_output'] = [pred0_4, pred1_4, pred2_s4, pred0_s3, predmid_s3, pred1_s3_up, pred0_s2, predmid_s2, pred1_s2]    # (B,H,W)

            if not self.frozen_former:
                outputs['recon_loss'] = F.mse_loss(left, features_left['recon_img']) + F.mse_loss(right, features_right['recon_img'])
                outputs['init_cost_volume'] = [
                    build_gwc_norm_volume(features_left['concat_feature6'], features_right['concat_feature6'], left.shape[-1]//32, 1, "cosine"),
                    build_gwc_norm_volume(features_left['concat_feature5'], features_right['concat_feature5'], left.shape[-1]//16, 1, "cosine"),
                    build_gwc_norm_volume(features_left['concat_feature4'], features_right['concat_feature4'], left.shape[-1]// 8, 1, "cosine"),
                    build_gwc_norm_volume(features_left['concat_feature3'], features_right['concat_feature3'], left.shape[-1]// 4, 1, "cosine"),
                ]
            outputs['disparity'] = pred1_s2.unsqueeze(1)

        else:
            # pred2_s4 = F.upsample(pred2_s4 * 8, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)
            # pred2_s4 = torch.squeeze(pred2_s4, 1)
            #
            # pred1_s3_up = F.upsample(pred1_s3 * 4, [left.size()[2], left.size()[3]], mode='bilinear',
            #                          align_corners=True)
            # pred1_s3_up = torch.squeeze(pred1_s3_up, 1)

            pred1_s2 = F.upsample(pred1_s2 * 2, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)
            # pred1_s2 = torch.squeeze(pred1_s2, 1)

            outputs = {"disparity": pred1_s2}

        return outputs


def CFNet(cfg):
    # return Former(maxdisp=cfg['model']['max_disp'], concat_feature_channel=16)
    # return cfnet(cfg['model']['max_disp'], frozen_former=True)
    return cfnet(cfg['model']['max_disp'], frozen_former=False)
