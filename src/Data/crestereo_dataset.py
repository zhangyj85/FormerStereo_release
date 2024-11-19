"""
load crestereo dataset.
Assuming the inter. intra params. similar to ScenefLow
Author: zhangyj85
Date: 2022.10.20
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
import matplotlib.pyplot as plt
import copy


class CREStereoDataset(Dataset):
    def __init__(self, args, mode, split='train'):
        super(CREStereoDataset, self).__init__()
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
                                # augmentation.Resize(self.args['crop_size']),
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
        # return Image.open(filename).convert('RGB').convert('L').convert('RGB')

    def load_disp(self, filename):
        data = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        data = Image.fromarray(data.astype('float32') / 32, mode='F')
        return data

    def _load_data(self, idx):
        path_img1 = self.split_set[idx]['img1']
        path_img2 = self.split_set[idx]['img2']
        path_disp1 = self.split_set[idx]['disp1']
        path_disp2 = self.split_set[idx]['disp2']

        # 载入双目图像
        img1 = self.load_image(path_img1)       # 读取双目图像
        img2 = self.load_image(path_img2)
        disp1 = self.load_disp(path_disp1)      # 读取视差图
        disp2 = self.load_disp(path_disp2)

        # 由于该数据集没有提供内参和外参, 因此人为假定该参数与 Sceneflow 一致. 可能会出错
        baseline = 1.0                      # 近似值
        internal = np.array([[1050.0, 0.0,    479.5],
                             [0.0,    1050.0, 269.5],
                             [0.0,    0.0,    1.0]])
        K1, K2 = internal.copy(), internal.copy()

        # 在假定外参和内参的情况下, 计算深度
        dep1 = (K1[0, 2] - K2[0, 2]) + K1[0, 0] * baseline / (np.array(disp1) + 1e-8)
        dep1 = dep1 * (np.array(disp1) > 1e-3).astype(dep1.dtype)
        dep1 = Image.fromarray(dep1)

        dep2 = (K1[0, 2] - K2[0, 2]) + K1[0, 0] * baseline / (np.array(disp2) + 1e-8)
        dep2 = dep2 * (np.array(disp2) > 1e-3).astype(dep2.dtype)
        dep2 = Image.fromarray(dep2)

        return {"img_left": img1, "img_right": img2,
                "gt_disp_left":disp1, "gt_disp_right": disp2,
                "gt_dep_left": dep1, "gt_dep_right": dep2,
                "mask_left":None, "mask_right":None,
                "K1": torch.from_numpy(K1), "K2": torch.from_numpy(K2), "b":baseline}

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
        gt1 = TF.to_tensor(np.array(one_sample['gt_disp_left']))       # to_tensor(), np类 -> tensor类, 不改变数值范围
        gt2 = TF.to_tensor(np.array(one_sample['gt_disp_right']))

        # mask1
        if one_sample["mask_left"] is not None:
            mask1 = TF.to_tensor(np.array(one_sample['mask_left']))
        else:
            mask1 = TF.to_tensor(np.ones((img1.shape[-2], img1.shape[-1])))

        # 将标定参数放到 tensor 中
        K1 = one_sample["K1"] #torch.from_numpy(one_sample["K1"])
        K2 = one_sample["K2"] #torch.from_numpy(one_sample["K2"])
        b  = torch.from_numpy(np.array(one_sample["b"]))

        return {"ir1":img1, "ir2":img2, "gt1":gt1,  "gt2":[], "mask1": mask1, "mask2": [],
                "K1": K1, "K2": K2, "b": b,
                "org1": TF.to_tensor(one_sample['org_left']) if self.mode == 'train' else img1,
                "org2": TF.to_tensor(one_sample['org_right']) if self.mode == 'train' else img2,
                }
