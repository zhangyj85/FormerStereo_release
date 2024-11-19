"""
从 ViT 推理得到 token, 并将 token 转化为图像 patch 格式
Dense ViT features proposed in "Vision Transformers for Dense Prediction"
Modified from: https://github.com/intel-isl/DPT
"""
import torch
import torch.nn as nn
import timm
import types
import math
import torch.nn.functional as F

import os
from utils.logger import warning

activations = {}    # 全局变量, 用来给 get_activation() 存放 hock 输出


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook


attention = {}      # 全局变量, 用来给 get_attention() 存放 hock 输出


def get_attention(name):
    def hook(module, input, output):
        x = input[0]
        B, N, C = x.shape
        qkv = (
            module.qkv(x)
            .reshape(B, N, 3, module.num_heads, C // module.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * module.scale

        attn = attn.softmax(dim=-1)  # [:,:,1,1:]
        attention[name] = attn

    return hook


def get_mean_attention_map(attn, token, shape):
    attn = attn[:, :, token, 1:]
    attn = attn.unflatten(2, torch.Size([shape[2] // 16, shape[3] // 16])).float()
    attn = torch.nn.functional.interpolate(
        attn, size=shape[2:], mode="bicubic", align_corners=False
    ).squeeze(0)

    all_attn = torch.mean(attn, 0)

    return all_attn


class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index :]


class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)


class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)

        return self.project(features)


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


# vit 前向推理, 并将中间层结果转化为图像, 然后输出
def forward_vit(pretrained, x):
    """
    pretrained: including model & post-process
    x: input in image shape: b, 3, h, w
    """
    b, c, h, w = x.shape

    # ViT 前向推理, glob 为x对应的最终特征, (B,N,C)
    pretrained.model.eval()     # 冻结 dropout & drop path
    with torch.no_grad():
        glob = pretrained.model.forward_flex(x)

    # 获取中间层特征, (B,N,C), layer_1 对应浅层特征, layer_4 为最深层特征
    layer_1 = pretrained.activations["1"]
    layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]
    layer_4 = pretrained.activations["4"]

    # 处理 [cls] token, 然后 (B,N,C) -> (B,C,N)
    layer_1 = pretrained.act_postprocess1[0:2](layer_1)
    layer_2 = pretrained.act_postprocess2[0:2](layer_2)
    layer_3 = pretrained.act_postprocess3[0:2](layer_3)
    layer_4 = pretrained.act_postprocess4[0:2](layer_4)

    # 将 token 转为图像, 得到的 layer 特征统一为 1/16, c=Dim
    unflatten = nn.Unflatten(
            2,
            torch.Size(
                [
                    h // pretrained.model.patch_size[1],
                    w // pretrained.model.patch_size[0],
                ]
            ),
        )

    if layer_1.ndim == 3:
        layer_1 = unflatten(layer_1)
    if layer_2.ndim == 3:
        layer_2 = unflatten(layer_2)
    if layer_3.ndim == 3:
        layer_3 = unflatten(layer_3)
    if layer_4.ndim == 3:
        layer_4 = unflatten(layer_4)

    return layer_1, layer_2, layer_3, layer_4


# position embedding, 根据输入进行 resize
# def _resize_pos_embed(self, posemb, gs_h, gs_w):
#     posemb_tok, posemb_grid = (
#         posemb[:, : self.start_index],
#         posemb[0, self.start_index :],
#     )
#
#     gs_old = int(math.sqrt(len(posemb_grid)))
#
#     posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
#     posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
#     posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
#
#     posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
#
#     return posemb


def _resize_pos_embed(self, posemb, gs_h, gs_w):
    # 使用 dinov2 的小改进
    posemb_tok, posemb_grid = (
        posemb[:, : self.start_index],
        posemb[0, self.start_index :],
    )

    gs_old = int(math.sqrt(len(posemb_grid)))
    if gs_old**2 == gs_h*gs_w and gs_h == gs_w:     # 和训练情况一致, 无需插值
        return posemb

    h_scale = (gs_h + 0.1) / math.sqrt(len(posemb_grid))
    w_scale = (gs_w + 0.1) / math.sqrt(len(posemb_grid))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid,
        scale_factor=(h_scale, w_scale),
        mode="bicubic"
    )
    assert gs_h == posemb_grid.shape[-2] and gs_w == posemb_grid.shape[-1]
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb


def forward_flex(self, x):
    b, c, h, w = x.shape

    # 对输入进行 position embedding (适应任意方形图像)
    pos_embed = self._resize_pos_embed(
        self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
    )

    B = x.shape[0]

    # 部分网络对 patch embedding 采用了 resnet
    if hasattr(self.patch_embed, "backbone"):
        x = self.patch_embed.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

    patch_W, patch_H = self.patch_size
    assert x.shape[2] % patch_H == 0, f"Input image height {x.shape[2]} is not a multiple of patch height {patch_H}"
    assert x.shape[3] % patch_W == 0, f"Input image width {x.shape[3]} is not a multiple of patch width: {patch_W}"
    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)

    # 将 [cls] token 加入到 token 序列中
    if getattr(self, "dist_token", None) is not None:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
    else:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

    x = x + pos_embed
    if hasattr(self, "pos_drop"):   # dinov2 没有 pos_drop
        x = self.pos_drop(x)

    for blk in self.blocks:
        x = blk(x)

    x = self.norm(x)

    return x


def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == "ignore":
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == "add":
        readout_oper = [AddReadout(start_index)] * len(features)
    elif use_readout == "project":
        readout_oper = [
            ProjectReadout(vit_features, start_index) for out_feat in features
        ]
    else:
        assert (
            False
        ), "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"

    return readout_oper


# 对预定义的网络增加中间输出节点, 并将 token unflatten 成为图像
def _make_vit_b16_backbone(
    model,
    psize=[16, 16],                     # patch size
    hooks=[2, 5, 8, 11],                # 在 hocks 的位置定义输出
    vit_features=768,                   # transformer blocks 的 dim
    use_readout="ignore",               # 如何处理 [cls] 与其他 token 的关系
    start_index=1,                      # [cls] 起始位置
    enable_attention_hooks=False,
):
    pretrained = nn.Module()

    # 将模型传给 pretrained 模型容器, 并插入输出层
    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    if enable_attention_hooks:
        # 这个 hock 应该是用来可视化看网络关注了那些地方?
        pretrained.model.blocks[hooks[0]].attn.register_forward_hook(
            get_attention("attn_1")
        )
        pretrained.model.blocks[hooks[1]].attn.register_forward_hook(
            get_attention("attn_2")
        )
        pretrained.model.blocks[hooks[2]].attn.register_forward_hook(
            get_attention("attn_3")
        )
        pretrained.model.blocks[hooks[3]].attn.register_forward_hook(
            get_attention("attn_4")
        )
        pretrained.attention = attention

    # 获取 [cls] 的操作
    readout_oper = get_readout_oper(vit_features, [vit_features]*4, use_readout, start_index)

    # 将 token 转为 FPN 特征
    pretrained.act_postprocess1 = nn.Sequential(
        readout_oper[0],                                                            # ignore [cls] token
        Transpose(1, 2),                                                            # BNC -> BCN
    )   # layer1, 1/4

    pretrained.act_postprocess2 = nn.Sequential(
        readout_oper[1],
        Transpose(1, 2),
    )   # layer2, 1/4

    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
    )   # layer3, 1/4

    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
    )   # layer4, 1/4

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = psize
    pretrained.model.vit_features = vit_features

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained


# 构造适配特定参数的网络结构
def _make_pretrained_dinov2_vitg14_518(
    use_readout="ignore",
    hooks=None,
    enable_attention_hooks=False
):
    # DINOv2 = vit-large, 14*14, 518*518
    repo_path = 'facebookresearch_dinov2_main'
    model = torch.hub.load(
        repo_or_dir=repo_path,
        model='dinov2_vitg14',
        pretrained=False,  # dinov2的接口
        source='local',
    )
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    path = "fundation_model_weights/dinov2_vitg14_pretrain.pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    else:
        warning(f"No valid DINOv2 pretrained model in path: {path}")

    hooks = [5, 11, 17, 23] if hooks == None else hooks
    return _make_vit_b16_backbone(
        model,
        psize=[model.patch_size]*2,
        hooks=hooks,
        vit_features=model.embed_dim,       # vit transformer 的通道数
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )


def _make_pretrained_dinov2_vitl14_518(
    use_readout="ignore",
    hooks=None,
    enable_attention_hooks=False
):
    # DINOv2 = vit-large, 14*14, 518*518
    repo_path = 'facebookresearch_dinov2_main'
    model = torch.hub.load(
        repo_or_dir=repo_path,
        model='dinov2_vitl14',
        pretrained=False,  # dinov2的接口
        source='local',
    )
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    path = "fundation_model_weights/dinov2_vitl14_pretrain.pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    else:
        warning(f"No valid DINOv2 pretrained model in path: {path}")

    hooks = [5, 11, 17, 23] if hooks == None else hooks
    return _make_vit_b16_backbone(
        model,
        psize=[model.patch_size]*2,
        hooks=hooks,
        vit_features=model.embed_dim,       # vit transformer 的通道数
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )


def _make_pretrained_dinov2_vitb14_518(
    use_readout="ignore",
    hooks=None,
    enable_attention_hooks=False
):
    repo_path = 'facebookresearch_dinov2_main'
    model = torch.hub.load(
        repo_or_dir=repo_path,
        model='dinov2_vitb14',
        pretrained=False,  # dinov2的接口
        source='local',
    )
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    path = "fundation_model_weights/dinov2_vitb14_pretrain.pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    else:
        warning(f"No valid DINOv2 pretrained model in path: {path}")

    hooks = [2, 5, 8, 11] if hooks == None else hooks
    return _make_vit_b16_backbone(
        model,
        psize=[model.patch_size]*2,
        hooks=hooks,
        vit_features=model.embed_dim,       # vit transformer 的通道数
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )


def _make_pretrained_dinov2_vits14_518(
    use_readout="ignore",
    hooks=None,
    enable_attention_hooks=False
):
    repo_path = 'facebookresearch_dinov2_main'
    model = torch.hub.load(
        repo_or_dir=repo_path,
        model='dinov2_vits14',
        pretrained=False,  # dinov2的接口
        source='local',
    )
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    path = "fundation_model_weights/dinov2_vits14_pretrain.pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    else:
        warning(f"No valid DINOv2 pretrained model in path: {path}")

    hooks = [1, 3, 5, 7] if hooks == None else hooks
    return _make_vit_b16_backbone(
        model,
        psize=[model.patch_size]*2,
        hooks=hooks,
        vit_features=model.embed_dim,       # vit transformer 的通道数
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )


def _make_pretrained_depth_anything_vitl14_518(
    use_readout="ignore",
    hooks=None,
    enable_attention_hooks=False
):
    # 加载模型 backbone
    repo_path = 'facebookresearch_dinov2_main'
    model = torch.hub.load(
        repo_or_dir=repo_path,
        model='dinov2_vitl14',
        pretrained=False,  # dinov2的接口
        source='local',
    )
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    # 加载 depth anything 权重 (DAM 和 DINOv2 同 encoder)
    path = "fundation_model_weights/depth_anything_vitl14.pth"
    if os.path.exists(path):
        DAM_weights = torch.load("fundation_model_weights/depth_anything_vitl14.pth", map_location="cpu")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in DAM_weights.items():
            if "depth_head." in k:
                # 排除无关权重
                continue
            prefix = "pretrained."
            name = k[len(prefix):] if prefix in k else k
            new_state_dict[name] = v

        # 对比发现, depth anything 的权重和 DINOv2 很像, 误差接近零
        model.load_state_dict(new_state_dict, strict=True)
    else:
        warning(f"No valid DAM pretrained model in path: {path}")

    hooks = [1, 3, 5, 7] if hooks == None else hooks
    return _make_vit_b16_backbone(
        model,
        psize=[model.patch_size] * 2,
        hooks=hooks,
        vit_features=model.embed_dim,  # vit transformer 的通道数
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )
