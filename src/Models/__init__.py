"""
Description:
(1) 构建算法总体框架，包括输入双目图像的预处理、选用匹配网络、视差结果后处理
(2) input: stereo images
    output: disparity map & confidence (options)
(3) This file including some utils funcs
"""
import importlib
import math
import torch.nn as nn
import torch.nn.functional as F


# 有效的可使用模型, 由于不同模型需要的环境不一样, 因此确定模型后再导入选定模型
__MODELS__ = ["FormerStereo", "Former_PSMNet", "Former_GwcNet", "Former_CFNet", "Former_RAFT"]


# 导入模型, 格式: 文件夹 + 模型 (两者同名)
def import_model(model_name: str):
    # load the module, will raise ImportError if module cannot be loaded
    if model_name not in __MODELS__:
        raise ValueError(
            f"Model {model_name} not in MODELS list. Valid models are {__MODELS__}"
        )
    m = importlib.import_module("Models." + model_name)     # 得到Models文件夹下的module文件
    return getattr(m, model_name)                           # 得到class的定义


class IMG_Processer(object):
    def __init__(self, d_rate):
        # padding方式
        self.rate = d_rate          # 最大降采样倍率

    def padding(self, img):
        # 图像padding, 左边和上边做padding
        _, _, H, W = img.shape
        top_pad = math.ceil(H / self.rate) * self.rate - H
        right_pad = math.ceil(W / self.rate) * self.rate - W
        # Following GwcNet, pad zeros on the top and the right side of the images
        # Improving: top reflect for context preserve, right zero pad for imambugurous matching
        self.size = (0, right_pad, top_pad, 0)
        img = F.pad(img, (0, 0, top_pad, 0), mode="reflect")        # top pad
        img = F.pad(img, (0, right_pad, 0, 0), mode="constant")     # right pad
        return img

    def unpadding(self, img):
        # 去除padding区域
        _, right_pad, top_pad, _ = self.size
        img = img[:, :, top_pad:, :img.shape[-1]-right_pad]
        return img

    def resize(self, img):
        # 将图像 resize 到合适的尺寸进行推理
        _, _, H, W = img.shape
        self.size = (H, W)
        resize_h = math.ceil(H / self.rate) * self.rate
        resize_w = math.ceil(W / self.rate) * self.rate
        img = F.interpolate(img, size=(resize_h, resize_w), mode="bilinear", align_corners=True)
        return img

    def restore(self, disp):
        _, _, h, w = disp.shape
        disp = F.interpolate(disp, size=self.size, mode="bilinear", align_corners=True)
        disp = disp * (self.size[-1] / w)
        return disp


def lcm(x, y):
    xy_gcd = math.gcd(x, y)
    xy_lcm = x * y // xy_gcd
    return xy_lcm



__DownRate__ = {
    "FormerStereo": lcm(28, 32),
    "Former_PSMNet": lcm(28, 16),# sam is lcm(32, 16)
    "Former_GwcNet": lcm(28, 16),
    "Former_CFNet": lcm(28, 32),
    "Former_RAFT": lcm(28, 16),
}


# 总模型定义
class MODEL(nn.Module):
    def __init__(self, config):
        super(MODEL, self).__init__()
        self.config = config
        model_name = config['model']['name']
        self.backbone = import_model(model_name)(config)
        self.processer = IMG_Processer(d_rate=__DownRate__[model_name])  # TODO: 将最大下采样与模型绑定

    def forward(self, imgL, imgR):

        if self.training:
            output = self.backbone(imgL, imgR)

        else:
            # 输入图像预处理, padding以适合网络
            imgL = self.processer.resize(imgL)
            imgR = self.processer.resize(imgR)
            output = self.backbone(imgL, imgR)

            # 对 padding 区域进行后处理
            output['disparity'] = self.processer.restore(output['disparity'])
            # TODO: 增加视差后处理模块(基于交叉熵的置信度估计滤波)

        return output


class LOSS(nn.Module):
    def __init__(self, config):
        super(LOSS, self).__init__()
        self.config = config
        model_name = config['model']['name']
        # 动态导入对应模型的损失函数模块
        m = importlib.import_module("Models." + model_name)
        loss_module = getattr(m, "loss_func")
        self.loss = loss_module(config)

    def forward(self, data_batch, training_output):
        return self.loss(data_batch, training_output)
