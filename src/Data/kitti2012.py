"""
load kitti2015 dataset.
Author: zhangyj85
Date: 2022.10.28
E-mail: zhangyj85@mail2.sysu.edu.cn
"""

import os
import re
import cv2
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from . import augmentation
# import matplotlib.pyplot as plt
import copy


class KITTIStereo2012(Dataset):
    def __init__(self, args, mode, split='train'):
        super(KITTIStereo2012, self).__init__()
        self.args = args["data"]
        self.mode = mode
        if mode not in ['train', 'test']:
            raise NotImplementedError

        with open(self.args['json_file']) as data_file:
            dataset = json.load(data_file)
            self.split_set = dataset[split]

        if self.args['augment']:
            self.augmentation = augmentation.Compose([
                                ### geometric augmentation
                                # augmentation.Resize(self.args['crop_size']),      # 由于真值不是稠密的, 无法 resize
                                # augmentation.VerticalFlip(),
                                # augmentation.RandomVdisp(),
                                augmentation.RandomCrop(self.args['crop_size'], shift=False,
                                                        max_disp=args["model"]["max_disp"],
                                                        min_disp=args["model"]["min_disp"]),
                                ### photometric augmentation
                                augmentation.ColorJitters(),
                                augmentation.ColorConvert(),
                                # augmentation.Reflection(self.args['crop_size']),
                                augmentation.RandomOcclusion()
                                ], height=self.args['crop_size'][0], width=self.args['crop_size'][1])
        else:
            self.augmentation = augmentation.Compose([augmentation.RandomCrop(self.args['crop_size'], shift=False)])

    def __len__(self):
        return len(self.split_set)

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_mask(self, filename):
        data = Image.open(filename)
        data = np.array(data).astype('float32') / 256
        data = Image.fromarray((data > 0.).astype('float32'), mode='F')
        return data

    def load_disp(self, filename):
        data = Image.open(filename)
        data = Image.fromarray(np.array(data).astype('float32') / 256, mode='F')
        return data

    def load_calib(self, filename):
        data = {}
        with open(filename, 'r') as f:
            for line in f.readlines():
                key, val = line.split(":", 1)   # 分割 key: value
                try:
                    data[key] = np.array([float(x) for x in val.split()])
                except ValueError:
                    pass
        return data

    def _load_data(self, idx):
        path_img1 = self.split_set[idx]['img1']
        path_img2 = self.split_set[idx]['img2']
        path_calib = self.split_set[idx]['calib']

        # 载入双目图像
        img1 = self.load_image(path_img1)       # 读取双目图像
        img2 = self.load_image(path_img2)

        # 载入标定参数
        calib = self.load_calib(path_calib)
        P_rect_02 = np.reshape(calib['P2'], (3, 4))      # 相机 02 的投影矩阵
        P_rect_03 = np.reshape(calib['P3'], (3, 4))      # 相机 03 的投影矩阵
        baseline = P_rect_02[0, 3] / P_rect_02[0, 0] - P_rect_03[0, 3] / P_rect_03[0, 0]
        K1 = P_rect_02[:, :3]   # 获取极线校正后的 K 矩阵 (原来的calib['K_02']是没有经过极线校正的)
        K2 = P_rect_03[:, :3]

        # 获取视差, 计算深度
        if 'disp' in self.split_set[idx].keys():
            path_disp = self.split_set[idx]['disp']
            disp = self.load_disp(path_disp)  # 读取视差图
            dep = (K1[0, 2] - K2[0, 2]) + K1[0, 0] * baseline / (np.array(disp) + 1e-8)
            dep = dep * (np.array(disp) > 1e-3).astype(dep.dtype)
            dep = Image.fromarray(dep)
        else:
            disp, dep = None, None

        try:
            path_mask = self.split_set[idx]['mask']
            mask = self.load_mask(path_mask)
        except:
            mask = Image.fromarray(np.zeros_like(np.array(disp)))

        if self.mode=="test":       # 去除错误的伪标签
            disp = np.array(disp)
            disp[:128, :] = 0.
            disp = Image.fromarray(disp)

        return {"img_left": img1, "img_right": img2,
                "gt_disp_left":disp, "gt_disp_right": None,
                "gt_dep_left": dep, "gt_dep_right": None,
                "mask_left":mask, "mask_right":None,
                "K1": torch.from_numpy(K1), "K2": torch.from_numpy(K2), "b":baseline,
                "ref_path": path_img1}

    def __getitem__(self, idx):

        one_sample = self._load_data(idx)       # 加载数据
        one_sample["org_left"] = copy.deepcopy(one_sample['img_left'])
        one_sample["org_right"] = copy.deepcopy(one_sample['img_left'])

        # 是否数据曾广
        if self.mode == 'train':
            one_sample = self.augmentation(one_sample)

        # 将双目图像载入tensor
        img1 = TF.to_tensor(one_sample['img_left'])                 # to_tensor(), PIL类 -> tensor类, 0-255 -> 0-1
        img2 = TF.to_tensor(one_sample['img_right'])

        # 依分布归一化到(-1,1)
        img1 = TF.normalize(img1, self.args['mean'], self.args['std'], inplace=True)
        img2 = TF.normalize(img2, self.args['mean'], self.args['std'], inplace=True)

        # ground truth 载入tensor, 无须归一化
        if one_sample['gt_disp_left'] is not None:
            gt1 = TF.to_tensor(np.array(one_sample['gt_disp_left']))       # to_tensor(), np类 -> tensor类, 不改变数值范围
        else:
            gt1 = TF.to_tensor(np.zeros((img1.shape[-2], img1.shape[-1])))

        # mask1
        if one_sample["mask_left"] is not None:
            mask1 = TF.to_tensor(np.array(one_sample['mask_left']))
        else:
            mask1 = TF.to_tensor(np.zeros((img1.shape[-2], img1.shape[-1])))

        # 将标定参数放到 tensor 中
        K1 = one_sample["K1"] #torch.from_numpy(one_sample["K1"])
        K2 = one_sample["K2"] #torch.from_numpy(one_sample["K2"])
        b  = torch.from_numpy(np.array(one_sample["b"]))

        return {"ir1":img1, "ir2":img2, "gt1":gt1,  "gt2":[], "mask1": mask1, "mask2": [],
                "K1": K1, "K2": K2, "b": b,
                "org1": TF.to_tensor(one_sample['org_left']) if self.mode == 'train' else img1,
                "org2": TF.to_tensor(one_sample['org_right']) if self.mode == 'train' else img2,
                "ref_path": [] if self.mode=='train' else one_sample['ref_path']
                }
