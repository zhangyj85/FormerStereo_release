"""
从 ViT 推理得到 token, 并将 token 转化为图像 patch 格式
Dense ViT features proposed in "Vision Transformers for Dense Prediction"
Modified from: https://github.com/intel-isl/DPT
"""
import torch
import torch.nn as nn
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
def forward_sam(pretrained, x):
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
    layer_1 = pretrained.activations["1"].permute(0, 3, 1, 2)   # b,h,w,c -> b,c,h,w
    layer_2 = pretrained.activations["2"].permute(0, 3, 1, 2)
    layer_3 = pretrained.activations["3"].permute(0, 3, 1, 2)
    layer_4 = pretrained.activations["4"].permute(0, 3, 1, 2)

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
    _, pos_h, pos_w, embed_dim = posemb.shape
    if gs_h == pos_h and gs_w == pos_w:     # 和训练情况一致, 无需插值
        return posemb

    h_scale = (gs_h + 0.1) / pos_h
    w_scale = (gs_w + 0.1) / pos_w
    posemb_grid = posemb.permute(0, 3, 1, 2) # bchw
    posemb_grid = F.interpolate(
        posemb_grid,
        scale_factor=(h_scale, w_scale),
        mode="bicubic"
    )
    assert gs_h == posemb_grid.shape[-2] and gs_w == posemb_grid.shape[-1]
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h, gs_w, -1)

    return posemb_grid


def forward_flex(self, x):
    b, c, h, w = x.shape

    # 对输入数据进行归一化 (验算后发现归一化方式几乎一致, 因此该步骤可省略)
    # mean = torch.tensor([0.485, 0.456, 0.406]).to(x.device)
    # std  = torch.tensor([0.229, 0.224, 0.225]).to(x.device)
    # x = x * std[None, :, None, None] + mean[None, :, None, None]
    # x = (x * 255. - self.pixel_mean.to(x.device)) / self.pixel_std.to(x.device)

    # 对输入进行 position embedding (适应任意方形图像)
    pos_embed = self._resize_pos_embed(
        self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
    )

    patch_W, patch_H = self.patch_size
    assert x.shape[2] % patch_H == 0, f"Input image height {x.shape[2]} is not a multiple of patch height {patch_H}"
    assert x.shape[3] % patch_W == 0, f"Input image width {x.shape[3]} is not a multiple of patch width: {patch_W}"
    x = self.patch_embed.proj(x).permute(0, 2, 3, 1)

    if getattr(self, "pos_embed", None) is not None:
        x = x + pos_embed

    for blk in self.blocks:
        x = blk(x)

    x = self.neck(x.permute(0, 3, 1, 2))    # b,c,h,w

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

    pretrained.model.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).reshape(1, -1, 1, 1)
    pretrained.model.pixel_std  = torch.tensor([58.395,  57.12,  57.375]).reshape(1, -1, 1, 1)

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained


def _make_pretrained_sam_vitl16_1024(
    use_readout="ignore",
    hooks=None,
    enable_attention_hooks=False
):
    # DINOv2 = vit-large, 16*16, 1024*1024
    from functools import partial
    from .image_encoder import ImageEncoderViT
    # sam 通用参数
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    # sam-vit-large 参数
    encoder_embed_dim = 1024
    encoder_depth = 24
    encoder_num_heads = 16
    encoder_global_attn_indexes = [5, 11, 17, 23]
    model = ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    )

    path = "fundation_model_weights/sam_vit_l_0b3195.pth"
    if os.path.exists(path):
        weights = torch.load(path, map_location="cpu")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in weights.items():
            if "image_encoder." in k:
                prefix = "image_encoder."
                name = k[len(prefix):] if prefix in k else k
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
    else:
        warning(f"No valid SAM pretrained model in path: {path}")

    hooks = [5, 11, 17, 23] if hooks == None else hooks
    return _make_vit_b16_backbone(
        model,
        psize=[vit_patch_size]*2,
        hooks=hooks,
        vit_features=encoder_embed_dim,       # vit transformer 的通道数
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
        start_index=0,
    )
