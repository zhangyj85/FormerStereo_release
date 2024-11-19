"""
Description:
        Data augmentation for stereo matching, including photometric transform, spacial transform, et al.

input:  the known variables in prior including stereo pair and calibration parameters (focal-length, baseline, K)
        image in type PIL, including left & right image
target: the disparity and mask, et. al.
        image in type PIL. including disp & dep

ignore the ground truth map of right views
"""
from __future__ import division
import math
import numpy as np
import numbers
import pdb
import cv2
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import copy


class Compose(object):
    """ Composes several transforms together.
    """

    def __init__(self, transforms, width, height):
        self.transforms = transforms
        self.padding_sample = PaddingSample(width, height)

    def check_consistency(self, sample):
        # check the conversion whether correct
        # disp & dep in left view, PIL type
        # K1: left calib, 3*3, tensor
        # K2: right calib, 3*3
        # b: baseline
        disp, dep = np.array(sample["gt_disp_left"]), np.array(sample["gt_dep_left"])
        K1, K2, b = sample["K1"].data.numpy(), sample["K2"].data.numpy(), sample["b"]
        disp_ = (K1[0, 2] - K2[0, 2]) + K1[0, 0] * b / (dep + 1e-8)
        valid = (dep > 1e-3).astype(dep.dtype)
        diff = np.abs(disp - disp_)
        diff[(valid < 1)] = 0.
        # assert np.sum(diff) / np.sum(valid) < 1e-3 or np.percentile(diff, 97) < 1e-3, "Depth and disparity is not matching!"
        if not ((np.sum(diff) / np.sum(valid) < 1e-3 or np.percentile(diff, 97) < 1e-3)):
            print("Depth and disparity is not matching! Mask the invalid values to ZERO.")
            sample["gt_disp_left"] = Image.fromarray(disp * valid * (diff < 1e-3).astype(disp.dtype))
            sample["gt_dep_left"] = Image.fromarray(dep * valid * (diff < 1e-3).astype(dep.dtype))

    def __call__(self, sample, check_consistency=True):
        # TODO: 名字对应列表
        """
        keys: ["img_left", "img_right", "gt_disp_left", "gt_disp_right", "gt_dep_left", "gt_dep_right",
               "mask_left", "mask_right", "K1", "K2"]
        """
        # self.check_consistency(sample)
        sample = self.padding_sample(sample)
        for t in self.transforms:           # get transform
            sample = t(sample)
        if check_consistency:
            self.check_consistency(sample)
        return sample


class PaddingSample(object):
    def __init__(self, width, height):
        self.min_width = width
        self.min_height = height

    def __call__(self, sample):
        width, height = sample['img_left'].size     # 图像实际大小
        pad_w = max(0, self.min_width - width)
        pad_h = max(0, self.min_height - height)
        # 对图像的操作, 左右 padding 0, 上下重复
        for key in ['img_left', 'img_right']:
            img = np.array(sample[key])
            img = np.pad(img, ((0, 0), (0, pad_w), (0,0)), mode="constant")
            img = np.pad(img, ((pad_h, 0), (0, 0), (0, 0)), mode="reflect")
            sample[key] = Image.fromarray(img)

        # 对视差和 mask 的操作, pad 0
        for key in ['gt_disp_left', 'gt_dep_left', 'mask_left']:
            if sample[key] is not None:
                img = np.array(sample[key])
                img = np.pad(img, ((pad_h, 0), (0, pad_w)), mode="constant")
                sample[key] = Image.fromarray(img.astype('float32'), mode='F')

        return sample


class RandomCrop(object):
    """ Randomly crop images, and add the disparity shift augmentation
    Note that the disparity shift augmentation is incorrect when shift is large because we do not consider the occlusion
    """

    def __init__(self, size, shift=False, max_disp=256, min_disp=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size    # size = (H, W)

        self.shift = shift
        self.disp_max = max_disp
        self.disp_min = min_disp

    def __call__(self, sample):
        # 获取裁剪框大小, 裁剪框不能超过原图大小
        origin_w, origin_h = sample['img_left'].size  # 返回图像的大小, 不考虑通道数
        crop_h, crop_w = self.size
        assert (crop_h <= origin_h) and (crop_w <= origin_w), "Invalid cropped size!"

        # 随机起点
        w_start = np.random.randint(0, origin_w - crop_w + 1)
        h_start = np.random.randint(0, origin_h - crop_h + 1)

        # 获取右图裁剪框的偏置
        if self.shift:
            # 获得当前裁剪框对应的视差图 & 计算视差分布, 注意存在稀疏真值的情况
            crop_disp = np.array(sample['gt_disp_left'])[h_start:h_start + crop_h, w_start:w_start + crop_w, ...]
            crop_disp_amin = np.percentile(crop_disp[crop_disp > 0], 10)        # 获得10%最小视差的数值
            crop_disp_amin = max(crop_disp_amin, self.disp_min)                 # 获得左移下限
            crop_disp_amax = np.percentile(crop_disp[crop_disp > 0], 90)        # 获得90%最大视差的数值
            crop_disp_amax = min(crop_disp_amax, self.disp_max)                 # 获得右移上限

            # 获取随机偏移坐标, 假设 target view 左移为正方向, 则正方向的移动会减小视差值
            # 两大约束: 空间状态约束(裁剪框不能超出图像平面), 视差分布约束(保证视差分布不会超出候选视差范围)
            shift_max = w_start
            shift_max = min(shift_max, int(crop_disp_amin - self.disp_min))  # 保证移动之后最小视差大于等于有效最小视差
            shift_min = -1 * (origin_w - w_start - crop_w)
            shift_min = max(shift_min, int(crop_disp_amax - self.disp_max))  # 保证移动之后的最大视差小于有效最大视差
            w_shift = int(np.random.uniform(shift_min, shift_max))           # 移动整数个像素
        else:
            w_shift = 0

        # 对左视图进行裁剪
        sample['img_left'] = TF.crop(sample['img_left'], h_start, w_start, crop_h, crop_w)
        np_disp = np.array(TF.crop(sample['gt_disp_left'], h_start, w_start, crop_h, crop_w))
        np_mask = (np_disp > 0).astype(np_disp.dtype)                                           # 原视差大于0的地方, 视差才是有效的
        sample['gt_disp_left'] = (np_disp - w_shift) * np_mask                                  # 将无效的视差值 mask 掉
        sample['gt_disp_left'] = Image.fromarray(sample['gt_disp_left'])
        for key in ['gt_dep_left', 'mask_left']:
            # disparity shift 不影响深度和mask
            if sample[key] is not None:
                sample[key] = TF.crop(sample[key], h_start, w_start, crop_h, crop_w)

        # 对右视图进行裁剪
        sample['img_right'] = TF.crop(sample['img_right'], h_start, w_start - w_shift, crop_h, crop_w)
        if sample['gt_dep_right'] is not None:
            sample['gt_dep_right'] = TF.crop(sample['gt_dep_right'], h_start, w_start - w_shift, crop_h, crop_w)
        if sample['gt_disp_right'] is not None:
            np_disp = np.array(TF.crop(sample['gt_disp_right'], h_start, w_start - w_shift, crop_h, crop_w))
            np_mask = (np_disp > 0).astype(np_disp.dtype)
            sample['gt_disp_right'] = (np_disp - w_shift) * np_mask
            sample['gt_disp_right'] = Image.fromarray(sample['gt_disp_right'])
        sample['K2'][0,2] = sample['K2'][0,2] + w_shift

        # 裁剪后的图像, 备份, 用于后续的照度一致性检测
        sample['org_left'] = copy.deepcopy(sample['img_left'])
        sample['org_right'] = copy.deepcopy(sample['img_right'])

        return sample


class Reflection(object):
    """模拟高斯光源, 以此来丰富光照带来的影响
    """

    def __init__(self, size=(256, 256), max_disp=256):
        self.size = size
        self.max_disp = max_disp

    def __call__(self, sample):
        if np.random.binomial(1, 0.5):
            return self.add_reflection(sample)
        else:
            return sample

    # generate gaussian windows, 数值范围(0,1)
    def gaussian(self, ksize=[512,512], shift=[0,0], sigma=[1,1]):
        """
        ksize: [height, width]
        sigma: [vertical direction sigma, horizon direction sigma], standard deviation
        shift: [center_h, center_w], center为零表示高斯核的中心在 ksize 的中心, 负数向左上移动, 正数向右下移动, guassian mean
        output: a guassian figure with h=height, w=width
        """
        sigma = np.array(sigma) * np.array(ksize)       # 根据尺度进行归一化的 sigma
        shift = np.array(shift) * np.array(ksize)//2    # 根据尺度进行归一化的 shift(center mean), (-1,1)
        gauss_h = np.array([math.exp(-(x - ksize[0] // 2 - shift[0]) ** 2 / float(2 * sigma[0] ** 2)) for x in range(ksize[0])])   # 行高斯(竖直方向的一维高斯)
        gauss_w = np.array([math.exp(-(x - ksize[1] // 2 - shift[1]) ** 2 / float(2 * sigma[1] ** 2)) for x in range(ksize[1])])   # 列高斯(水平方向的一维高斯)
        gauss_h = gauss_h.reshape(ksize[0], 1)          # 转为列向量, (ksize[0],) -> (kisze[0], 1)
        gauss_w = gauss_w.reshape(1, ksize[1])          # 转为行向量, (ksize[0],) -> (1, ksize[1])
        gauss   = gauss_h @ gauss_w                     # 矩阵乘法,  (ksize[0], ksize[1])

        return gauss

    def add_reflection(self, sample):

        # reference 图像的高斯光源中心坐标 (归一化到 (-1, 1))
        nm = self.max_disp / self.size[1] * 2 - 1       # 归一化水平坐标最小值, [-1, 1]
        center_h = np.random.uniform(-1, 1)             # 随机光源垂直中心
        center_w = np.random.uniform(nm, 1)             # 随机光源水平中心
        sigma = np.random.uniform(0.05, 0.2, 2)         # 高斯光源的反射方差, 左右视图不相等
        mean_light = np.random.uniform(128, 192, 2)     # 高斯光源的平均亮度, 左右视图不相等

        # 对 target view 的光源中心进行水平扰动
        shift_w = np.random.uniform(-0.05, 0.05)
        target_w = center_w + shift_w

        light1 = (self.gaussian(ksize=self.size,
                                shift=[center_h, center_w], sigma=sigma) * mean_light[0])   # / mean_light[0]
        light2 = (self.gaussian(ksize=self.size,
                                shift=[center_h, target_w], sigma=sigma) * mean_light[1])   # / mean_light[1]

        re_size = self.size if len(np.array(sample['img_left']).shape)==2 else (self.size[0], self.size[1], 1)
        sample['img_left']  = Image.fromarray(np.uint8(np.clip(np.array(sample['img_left'])  + light1.reshape(re_size), 0, 255)))
        sample['img_right'] = Image.fromarray(np.uint8(np.clip(np.array(sample['img_right']) + light2.reshape(re_size), 0, 255)))

        return sample


class ColorJitters(object):
    """
    非对称的色彩扰动, 随机改变输入图像的亮度、对比图, 色调等
    超参设置参考AANet
    """

    def __init__(self, asymmetric=True):
        self.asymmetric = asymmetric

    def __call__(self, sample):
        # 照度的变化可以作为一个调参项
        random_brightness = np.random.uniform(0.5, 2.0, 2)  # 随机亮度变化
        random_gamma = np.random.uniform(0.7, 1.5, 2)       # 随机伽马变化
        random_contrast = np.random.uniform(0.8, 1.2, 2)    # 随机对比度变化
        random_hue = np.random.uniform(-0.1, 0.1, 2)        # 随机色调
        random_saturation = np.random.uniform(0.8, 1.2, 2)  # 随机色彩饱和度
        # 对左图应用某一随机变化
        sample['img_left'] = TF.adjust_brightness(sample['img_left'], random_brightness[0])
        sample['img_left'] = TF.adjust_gamma(sample['img_left'], random_gamma[0])
        sample['img_left'] = TF.adjust_contrast(sample['img_left'], random_contrast[0])
        sample['img_left'] = TF.adjust_hue(sample['img_left'], random_hue[0])
        sample['img_left'] = TF.adjust_saturation(sample['img_left'], random_saturation[0])

        # 对右图应用另一个随机变化
        if self.asymmetric:
            sample['img_right'] = TF.adjust_brightness(sample['img_right'], random_brightness[1])
            sample['img_right'] = TF.adjust_gamma(sample['img_right'], random_gamma[1])
            sample['img_right'] = TF.adjust_contrast(sample['img_right'], random_contrast[1])
            sample['img_right'] = TF.adjust_hue(sample['img_right'], random_hue[1])
            sample['img_right'] = TF.adjust_saturation(sample['img_right'], random_saturation[1])
        else:
            sample['img_right'] = TF.adjust_brightness(sample['img_right'], random_brightness[0])
            sample['img_right'] = TF.adjust_gamma(sample['img_right'], random_gamma[0])
            sample['img_right'] = TF.adjust_contrast(sample['img_right'], random_contrast[0])
            sample['img_right'] = TF.adjust_hue(sample['img_right'], random_hue[0])
            sample['img_right'] = TF.adjust_saturation(sample['img_right'], random_saturation[0])
        return sample


class ColorConvert(object):
    """
    非对称的通道变换, 将 RGB 随机变成 灰度图
    由于使用非对称会导致严重的性能下降, 因此这里使用对称的变换, 将一对 RGB 转换为 Gray
    """

    def __init__(self, convert_left=False):
        self.convert_left = convert_left

    def __call__(self, sample):
        # if self.convert_left and np.random.binomial(1, 0.5):
        #     # 由于左图需要大量语义信息以估计 invisible 区域, 因此对左图进行灰度化会严重匹配性能
        #     sample['img_left'] = sample['img_left'].convert('L')
        # if np.random.binomial(1, 0.5):
        #     sample['img_right'] = sample['img_right'].convert('L')
        if np.random.binomial(1, 0.5):
            sample['img_left'] = sample['img_left'].convert('L')
            sample['img_right'] = sample['img_right'].convert('L')

        sample['img_left'] = sample['img_left'].convert('RGB')
        sample['img_right'] = sample['img_right'].convert('RGB')
        return sample


class VerticalFlip(object):
    """
    对输入的样本进行垂直翻转(不改变视差关系和深度, 仅影响标定参数K)
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        if np.random.binomial(1, 0.5):
            for key in sample.keys():
                if isinstance(sample[key], Image.Image):
                    sample[key] = sample[key].transpose(Image.FLIP_TOP_BOTTOM)
            W, H = sample['img_left'].size
            sample['K1'][1, 2] = H - sample['K1'][1, 2]
            sample['K2'][1, 2] = H - sample['K2'][1, 2]
        return sample


class Resize(object):
    """
    对输入图像随机改变大小, 注意需要放在crop之前, 使每次输出网络的图像尺寸都是一致的
    由于Resize对于边缘没有很好的定义方案, 因此采用该方法会导致视差&深度一致性检查不通过, 需要参考其他算法进行改进
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size    # size = (H, W)

    def __call__(self, sample):
        if np.random.binomial(1, 0.5):
            origin_w, origin_h = sample['img_left'].size
            scale_min = max(self.size[0] / origin_h, self.size[1] / origin_w)   # 最大的缩放率, 将整张图像缩小并送入训练
            scale_max = 1.0
            resize_scale = np.random.uniform(scale_min, scale_max)

            # 对图像进行缩放
            left, right = np.array(sample['img_left']), np.array(sample['img_right'])
            left = cv2.resize(left, None, fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_CUBIC)
            right = cv2.resize(right, None, fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_CUBIC)
            sample['img_left'] = Image.fromarray(left)
            sample['img_right'] = Image.fromarray(right)

            # 对真值缩放
            disp = np.array(sample['gt_disp_left'])
            disp = cv2.resize(disp, None, fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_NEAREST) * resize_scale
            sample['gt_disp_left'] = Image.fromarray(disp)

            # 对其余输入进行同等操作
            for key in ["gt_dep_left", "mask_left"]:
                if sample[key] is not None:
                    value = np.array(sample[key])
                    value = cv2.resize(value, None, fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_NEAREST)
                    sample[key] = Image.fromarray(value)

            # 对标定参数进行修改
            sample["K1"] *= resize_scale
            sample["K1"][2, 2] = 1

            sample["K2"] *= resize_scale
            sample["K2"][2, 2] = 1
        return sample


class Blurry(object):
    def __init__(self):
        self.kernels = [np.ones((i,i), np.float32) / (i*i) for i in range(1,9+1,2)]

    def __call__(self, sample):
        if np.random.binomial(1, 0.5):
            idx = np.random.randint(5)  # 选择一个 kernel 进行模糊处理
            kernel = self.kernels[idx]
            img_left = cv2.filter2D(np.array(sample["img_left"]), -1, kernel)
            sample["img_left"] = Image.fromarray(img_left)
            img_rigth = cv2.filter2D(np.array(sample["img_right"]), -1, kernel)
            sample["img_right"] = Image.fromarray(img_rigth)

        return sample


class RandomVdisp(object):
    """
    Random vertical disparity augmentation of right image, geometric unsymmetric-augmentation
    """
    def __init__(self, angle=0.1, px=2):
        self.angle = angle
        self.px = px

    def __call__(self, sample):
        if np.random.binomial(1, 0.5):
            w, h = sample['img_right'].size
            right0 = np.array(sample['img_right'])

            px2 = np.random.uniform(-self.px, self.px)
            angle2 = np.random.uniform(-self.angle, self.angle)
            image_center = (np.random.uniform(0, h), np.random.uniform(0, w))
            rot_mat = cv2.getRotationMatrix2D(image_center, angle2, 1.0)        # [旋转中心, 旋转角度, 缩放比例]
            right1 = cv2.warpAffine(right0, rot_mat, (w, h), flags=cv2.INTER_CUBIC)
            trans_mat = np.float32([[1, 0, 0], [0, 1, px2]])                    # 生成垂直平移扰动
            right2 = cv2.warpAffine(right1, trans_mat, (w, h), flags=cv2.INTER_CUBIC)

            sample['img_right'] = Image.fromarray(right2)

        return sample


class RandomOcclusion(object):
    """
    randomly occlude a region of right image
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        if np.random.binomial(1, 0.5):
            w, h = sample['img_right'].size             # 返回当前图像的大小, 对任意通道的图像均有效
            sx = int(np.random.uniform(50, 150))
            sy = int(np.random.uniform(50, 150))
            cx = int(np.random.uniform(sx, h - sx))
            cy = int(np.random.uniform(sy, w - sy))
            img_right = np.array(sample['img_right'])
            # 根据图像均值填充遮挡区域的像素值
            img_right[cx - sx:cx + sx, cy - sy:cy + sy] = np.mean(np.mean(img_right, 0), 0)[np.newaxis, np.newaxis]
            sample['img_right'] = Image.fromarray(img_right)
        return sample
