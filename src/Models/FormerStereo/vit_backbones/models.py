import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .vit import (
    _make_pretrained_dinov2_vitg14_518,
    _make_pretrained_dinov2_vitl14_518,
    _make_pretrained_dinov2_vitb14_518,
    _make_pretrained_dinov2_vits14_518,
    _make_pretrained_depth_anything_vitl14_518,
    forward_vit,
)

from .sam_vit.vit import (
    _make_pretrained_sam_vitl16_1024,
    forward_sam,
)

from .swin_transformer import (
    _make_pretrained_swinmim_swin4_224,
    forward_swin,
)


hooks = {
    "dinov2_vitg14_518": [5, 11, 17, 23],
    "dinov2_vitl14_518": [5, 11, 17, 23],   # (24 // 4) * i - 1, i=1,2,3,4
    "dinov2_vitb14_518": [2, 5, 8, 11],     # (12 // 4) * i - 1
    "dinov2_vits14_518": [1, 3, 5, 7],      # ( 8 // 4) * i - 1
    "dam_vitl14_518": [5, 11, 17, 23],
    "sam_vitl16_1024": [5, 11, 17, 23],
}

forward_fn = {
    "dinov2_vitg14_518": forward_vit,
    "dinov2_vitl14_518": forward_vit,   # (24 // 4) * i - 1, i=1,2,3,4
    "dinov2_vitb14_518": forward_vit,     # (12 // 4) * i - 1
    "dinov2_vits14_518": forward_vit,      # ( 8 // 4) * i - 1
    "dam_vitl14_518": forward_vit,
    "sam_vitl16_1024": forward_sam,
}


def _make_encoder(
    backbone,
    hooks=None,
    use_readout="ignore",
    enable_attention_hooks=False,
):
    if backbone == "dinov2_vitl14_518":
        pretrained = _make_pretrained_dinov2_vitl14_518(
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
    elif backbone == "dinov2_vitb14_518":
        pretrained = _make_pretrained_dinov2_vitb14_518(
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
    elif backbone == "dinov2_vitg14_518":
        pretrained = _make_pretrained_dinov2_vitg14_518(
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
    elif backbone == "dinov2_vits14_518":
        pretrained = _make_pretrained_dinov2_vits14_518(
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
    elif backbone == "dam_vitl14_518":
        pretrained = _make_pretrained_depth_anything_vitl14_518(
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
    elif backbone == "sam_vitl16_1024":
        pretrained = _make_pretrained_sam_vitl16_1024(
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
    else:
        # TODO: 增加 ViT-g, ViT-s
        print(f"Backbone '{backbone}' not implemented")
        assert False

    return pretrained


class ViT_DenseDPT_v4_2(nn.Module):
    """DINOv2 + DPT, outputs in 1/4 scale
       use convex upsample for different scale features: 使用该上采样, 有效解决了DPT导致的棋盘格效应, 性能进一步提升
       use recon mlp
    """
    def __init__(
            self,
            backbone="dinov2_vitl14_518",
            features=[64, 96, 128, 256],
            out_feats=32,
            readout="ignore",
            enable_attention_hooks=False,
    ):
        super().__init__()

        # Instantiate backbone
        self.backbone = _make_encoder(
            backbone,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )
        self.forward_fn = forward_fn[backbone]
        self.pretrained_dict = copy.deepcopy(self.backbone.model.state_dict())
        vit_features = self.backbone.model.vit_features

        # follow DPT to acquire multi-scale features
        from Models.FormerStereo.submodule import learnable_upsample
        self.scale_fn1 = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=vit_features),
            learnable_upsample(in_chans=vit_features, out_chans=features[0], rate=4),
            nn.Conv2d(features[0], features[0], kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        )
        self.scale_fn2 = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=vit_features),
            learnable_upsample(in_chans=vit_features, out_chans=features[1], rate=2),
            nn.Conv2d(features[1], features[1], kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        )
        self.scale_fn3 = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=vit_features),
            learnable_upsample(in_chans=vit_features, out_chans=features[2], rate=1),
            nn.Conv2d(features[2], features[2], kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        )
        self.scale_fn4 = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=vit_features),
            learnable_upsample(in_chans=vit_features, out_chans=features[3], rate=1),
            nn.Conv2d(features[3], features[3], kernel_size=3, stride=2, padding=1, bias=False, groups=1),
        )

        # fusion multi-scale features
        from Models.FormerStereo.vit_backbones.DPT_Fusion import FeatureFusionBlock_custom
        self.fuse_layer4 = FeatureFusionBlock_custom(features[3], nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True, res1=False)
        self.fuse_layer3 = FeatureFusionBlock_custom(features[2], nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)
        self.fuse_layer2 = FeatureFusionBlock_custom(features[1], nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)
        self.fuse_layer1 = FeatureFusionBlock_custom(features[0], nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)

        # feature matching mlp
        self.head = nn.Sequential(
            nn.Conv2d(features[0], 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=256),
            nn.GELU(),
            nn.Conv2d(256, out_feats, kernel_size=3, stride=1, padding=1, bias=False),
        )

        # reconstruction mlp
        self.recon_head = nn.Sequential(
            nn.Conv2d(features[0], 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=256),
            learnable_upsample(in_chans=256, out_chans=3, rate=4),
        )

    def weight_load(self):
        # 加载 DINOv2 特征提取网络的官方预训练权重
        self.backbone.model.load_state_dict(self.pretrained_dict, strict=True)
        # 冻结 ViT-L/16 的参数, 注意 ViT 定义在 backbone.model 中, 而 backbone 本身包含我自定义的网络子结构
        for p in self.backbone.model.parameters():
            p.requires_grad = False
        self.backbone.model.eval()

    def forward_one(self, x):
        _, _, H, W = x.shape
        layer_1, layer_2, layer_3, layer_4 = self.forward_fn(self.backbone, x)  # 1/14
        layer_1 = self.scale_fn1(layer_1)                                   # 1/3.5
        layer_2 = self.scale_fn2(layer_2)                                   # 1/7
        layer_3 = self.scale_fn3(layer_3)                                   # 1/14
        layer_4 = self.scale_fn4(layer_4)                                   # 1/28

        layer_4 = self.fuse_layer4(layer_4)                                 # 1/14
        layer_3 = self.fuse_layer3(layer_4, layer_3)                        # 1/7
        layer_2 = self.fuse_layer2(layer_3, layer_2)                        # 1/3.5
        layer_1 = self.fuse_layer1(layer_2, layer_1)                        # 1/1.75

        y = F.interpolate(layer_1, size=(H // 4, W // 4), mode="bilinear")
        feats_y = self.head(y)
        recon_y = self.recon_head(y)

        return feats_y, recon_y

    def forward(self, left, right):
        left_features, recon_left = self.forward_one(left)
        right_features, recon_right = self.forward_one(right)
        return left_features, right_features, recon_left, recon_right


class Swin_DenseDPT_v4_2(nn.Module):
    """DINOv2 + DPT, outputs in 1/4 scale
       use convex upsample for different scale features: 使用该上采样, 有效解决了DPT导致的棋盘格效应, 性能进一步提升
       use recon mlp
    """
    def __init__(
            self,
            backbone="dinov2_vitl14_518",
            features=[64, 96, 128, 256],
            out_feats=32,
            readout="ignore",
            enable_attention_hooks=False,
    ):
        super().__init__()

        # Instantiate backbone
        self.backbone = _make_pretrained_swinmim_swin4_224(
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )
        self.pretrained_dict = copy.deepcopy(self.backbone.model.state_dict())
        vit_features = self.backbone.model.vit_features

        # follow DPT to acquire multi-scale features
        from Models.FormerStereo.submodule import learnable_upsample
        self.scale_fn1 = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=vit_features * 1),
            learnable_upsample(in_chans=vit_features * 1, out_chans=features[0], rate=1),
            nn.Conv2d(features[0], features[0], kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        )
        self.scale_fn2 = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=vit_features * 2),
            learnable_upsample(in_chans=vit_features * 2, out_chans=features[1], rate=1),
            nn.Conv2d(features[1], features[1], kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        )
        self.scale_fn3 = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=vit_features * 4),
            learnable_upsample(in_chans=vit_features * 4, out_chans=features[2], rate=1),
            nn.Conv2d(features[2], features[2], kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        )
        self.scale_fn4 = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=vit_features * 8),
            learnable_upsample(in_chans=vit_features * 8, out_chans=features[3], rate=1),
            nn.Conv2d(features[3], features[3], kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        )

        # fusion multi-scale features
        from Models.FormerStereo.vit_backbones.DPT_Fusion import FeatureFusionBlock_custom
        self.fuse_layer4 = FeatureFusionBlock_custom(features[3], nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True, res1=False)
        self.fuse_layer3 = FeatureFusionBlock_custom(features[2], nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)
        self.fuse_layer2 = FeatureFusionBlock_custom(features[1], nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)
        self.fuse_layer1 = FeatureFusionBlock_custom(features[0], nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)

        # feature matching mlp
        self.head = nn.Sequential(
            nn.Conv2d(features[0], 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=256),
            nn.GELU(),
            nn.Conv2d(256, out_feats, kernel_size=3, stride=1, padding=1, bias=False),
        )

        # reconstruction mlp
        self.recon_head = nn.Sequential(
            nn.Conv2d(features[0], 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=256),
            learnable_upsample(in_chans=256, out_chans=3, rate=4),
        )

    def weight_load(self):
        # 加载 DINOv2 特征提取网络的官方预训练权重
        self.backbone.model.load_state_dict(self.pretrained_dict, strict=True)
        # 冻结 ViT-L/16 的参数, 注意 ViT 定义在 backbone.model 中, 而 backbone 本身包含我自定义的网络子结构
        for p in self.backbone.model.parameters():
            p.requires_grad = False
        self.backbone.model.eval()

    def forward_one(self, x):
        _, _, H, W = x.shape
        layer_1, layer_2, layer_3, layer_4 = forward_swin(self.backbone, x) # [1/4, 1/8, 1/16, 1/32]
        layer_1 = self.scale_fn1(layer_1)                                   # 1/4
        layer_2 = self.scale_fn2(layer_2)                                   # 1/8
        layer_3 = self.scale_fn3(layer_3)                                   # 1/16
        layer_4 = self.scale_fn4(layer_4)                                   # 1/32

        layer_4 = self.fuse_layer4(layer_4)                                 # 1/16
        layer_3 = self.fuse_layer3(layer_4, layer_3)                        # 1/8
        layer_2 = self.fuse_layer2(layer_3, layer_2)                        # 1/4
        layer_1 = self.fuse_layer1(layer_2, layer_1)                        # 1/2

        y = F.interpolate(layer_1, size=(H // 4, W // 4), mode="bilinear")
        feats_y = self.head(y)
        recon_y = self.recon_head(y)

        return feats_y, recon_y

    def forward(self, left, right):
        left_features, recon_left = self.forward_one(left)
        right_features, recon_right = self.forward_one(right)
        return left_features, right_features, recon_left, recon_right


class Swin_Simple_v4_2(nn.Module):
    """DINOv2 + DPT, outputs in 1/4 scale
       use convex upsample for different scale features: 使用该上采样, 有效解决了DPT导致的棋盘格效应, 性能进一步提升
       use recon mlp
    """
    def __init__(
            self,
            backbone="dinov2_vitl14_518",
            features=[64, 96, 128, 256],
            out_feats=32,
            readout="ignore",
            enable_attention_hooks=False,
    ):
        super().__init__()

        # Instantiate backbone
        self.backbone = _make_pretrained_swinmim_swin4_224(
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )
        self.pretrained_dict = copy.deepcopy(self.backbone.model.state_dict())
        vit_features = self.backbone.model.vit_features

        # 使用 pixelshuffle 进行特征上采样
        self.scale_fn1 = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=vit_features * 1),
            nn.Conv2d(vit_features * 1, features[0], kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.Conv2d(features[0], out_feats * 1 * 1, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        )
        self.scale_fn2 = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=vit_features * 2),
            nn.Conv2d(vit_features * 2, features[1], kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.Conv2d(features[1], out_feats * 2 * 2, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.PixelShuffle(2),
        )
        self.scale_fn3 = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=vit_features * 4),
            nn.Conv2d(vit_features * 4, features[2], kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.Conv2d(features[2], out_feats * 4 * 4, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.PixelShuffle(4),
        )
        self.scale_fn4 = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=vit_features * 8),
            nn.Conv2d(vit_features * 8, features[3], kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.Conv2d(features[3], out_feats * 8 * 8, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.PixelShuffle(8),
        )

        # feature matching mlp
        self.head = nn.Sequential(
            nn.Conv2d(out_feats * 4, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=256),
            nn.GELU(),
            nn.Conv2d(256, out_feats, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def weight_load(self):
        # 加载 DINOv2 特征提取网络的官方预训练权重
        self.backbone.model.load_state_dict(self.pretrained_dict, strict=True)
        # 冻结 ViT-L/16 的参数, 注意 ViT 定义在 backbone.model 中, 而 backbone 本身包含我自定义的网络子结构
        for p in self.backbone.model.parameters():
            p.requires_grad = False
        self.backbone.model.eval()

    def forward_one(self, x):
        _, _, H, W = x.shape
        layer_1, layer_2, layer_3, layer_4 = forward_swin(self.backbone, x) # [1/4, 1/8, 1/16, 1/32]
        layer_1 = self.scale_fn1(layer_1)                                   # 1/4
        layer_2 = self.scale_fn2(layer_2)                                   # 1/8
        layer_3 = self.scale_fn3(layer_3)                                   # 1/16
        layer_4 = self.scale_fn4(layer_4)                                   # 1/32

        fuse_layer = torch.cat([layer_1, layer_2, layer_3, layer_4], dim=1)
        y = F.interpolate(fuse_layer, size=(H // 4, W // 4), mode="bilinear")
        feats_y = self.head(y)
        recon_y = None

        return feats_y, recon_y

    def forward(self, left, right):
        left_features, recon_left = self.forward_one(left)
        right_features, recon_right = self.forward_one(right)
        return left_features, right_features, recon_left, recon_right


class ViT_DenseDPT_v4_2_CFNet2(nn.Module):
    """DINOv2 + DPT, outputs in 1/4 scale
       use convex upsample for different scale features: 使用该上采样, 有效解决了DPT导致的棋盘格效应, 性能进一步提升
       use recon mlp
    """
    def __init__(
            self,
            backbone="dinov2_vitl14_518",
            features=[64, 96, 128, 256],
            out_feats=32,
            readout="ignore",
            enable_attention_hooks=False,
    ):
        super().__init__()

        # Instantiate backbone
        self.backbone = _make_encoder(
            backbone,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )
        self.pretrained_dict = copy.deepcopy(self.backbone.model.state_dict())
        vit_features = self.backbone.model.vit_features

        # follow DPT to acquire multi-scale features
        from Models.FormerStereo.submodule import learnable_upsample
        self.scale_fn1 = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=vit_features),
            learnable_upsample(in_chans=vit_features, out_chans=features[0], rate=4),
            nn.Conv2d(features[0], features[0], kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        )
        self.scale_fn2 = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=vit_features),
            learnable_upsample(in_chans=vit_features, out_chans=features[1], rate=2),
            nn.Conv2d(features[1], features[1], kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        )
        self.scale_fn3 = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=vit_features),
            learnable_upsample(in_chans=vit_features, out_chans=features[2], rate=1),
            nn.Conv2d(features[2], features[2], kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        )
        self.scale_fn4 = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=vit_features),
            learnable_upsample(in_chans=vit_features, out_chans=features[3], rate=1),
            nn.Conv2d(features[3], features[3], kernel_size=3, stride=2, padding=1, bias=False, groups=1),
        )

        # fusion multi-scale features
        from Models.FormerStereo.vit_backbones.DPT_Fusion import FeatureFusionBlock_custom
        self.fuse_layer4 = FeatureFusionBlock_custom(features[3], nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True, res1=False)
        self.fuse_layer3 = FeatureFusionBlock_custom(features[2], nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)
        self.fuse_layer2 = FeatureFusionBlock_custom(features[1], nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)
        self.fuse_layer1 = FeatureFusionBlock_custom(features[0], nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)

        # feature matching mlp
        self.head = nn.Sequential(
            nn.Conv2d(features[0], 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=256),
            nn.GELU(),
            nn.Conv2d(256, out_feats, kernel_size=3, stride=1, padding=1, bias=False),
        )

        # reconstruction mlp
        self.recon_head = nn.Sequential(
            nn.Conv2d(features[0], 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=256),
            learnable_upsample(in_chans=256, out_chans=3, rate=4),
        )

    def weight_load(self):
        # 加载 DINOv2 特征提取网络的官方预训练权重
        self.backbone.model.load_state_dict(self.pretrained_dict, strict=True)
        # 冻结 ViT-L/16 的参数, 注意 ViT 定义在 backbone.model 中, 而 backbone 本身包含我自定义的网络子结构
        for p in self.backbone.model.parameters():
            p.requires_grad = False
        self.backbone.model.eval()

    def forward_one(self, x):
        _, _, H, W = x.shape
        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.backbone, x)  # 1/14
        layer_1 = self.scale_fn1(layer_1)                                   # 1/3.5
        layer_2 = self.scale_fn2(layer_2)                                   # 1/7
        layer_3 = self.scale_fn3(layer_3)                                   # 1/14
        layer_4 = self.scale_fn4(layer_4)                                   # 1/28

        layer_4 = self.fuse_layer4(layer_4)                                 # 1/14
        layer_3 = self.fuse_layer3(layer_4, layer_3)                        # 1/7
        layer_2 = self.fuse_layer2(layer_3, layer_2)                        # 1/3.5
        layer_1 = self.fuse_layer1(layer_2, layer_1)                        # 1/1.75

        y = F.interpolate(layer_1, size=(H // 4, W // 4), mode="bilinear")
        feats_y = self.head(y)
        recon_y = self.recon_head(y)

        output = {}
        output['recon_img'] = recon_y
        output['feats_cat'] = feats_y
        output['layer_1'] = F.interpolate(layer_1, size=(H // 4, W // 4), mode="bilinear")
        output['layer_2'] = F.interpolate(layer_2, size=(H // 8, W // 8), mode="bilinear")
        output['layer_3'] = F.interpolate(layer_3, size=(H //16, W //16), mode="bilinear")
        output['layer_4'] = F.interpolate(layer_4, size=(H //32, W //32), mode="bilinear")

        return output

    def forward(self, left, right):
        left_outputs = self.forward_one(left)
        right_outputs = self.forward_one(right)
        return left_outputs, right_outputs

