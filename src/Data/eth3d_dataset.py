"""
Copy from CasStereo
Author: Yongjian Zhang
E-mail: zhangyj85@mail2.sysu.edu.cn
Date:   2022.04.15
"""
import os
import re
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from . import augmentation
# import matplotlib.pyplot as plt
import copy
import cv2


# read all lines in a file
def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]   # rstrip() 删除 string 字符串末尾的指定字符，默认为空白符，包括空格、换行符、回车符、制表符
    return lines


# read an .pfm file into numpy array, used to load SceneFlow disparity files
def pfm_imread(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


class ETH3DDataset(Dataset):
    def __init__(self, args, mode, split='train'):
        super(ETH3DDataset, self).__init__()
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

    def load_mask(self, filename):
        return Image.open(filename).convert('L')

    def load_disp(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        data = Image.fromarray(data.astype('float32'), mode='F')
        return data

    def load_calib(self, calib_path):
        lines = read_all_lines(calib_path)
        calib = {}
        for line in lines:
            key, val = line.split("=")
            try:
                calib[key] = float(val)
            except:
                val = val.replace(";", "").replace("[", "").replace("]", "")
                val = np.fromstring(val, dtype=float, sep=' ')
                calib[key] = np.reshape(val, [3, 3])
        return calib

    def _load_data(self, idx):
        path_img1 = self.split_set[idx]['img1']
        path_img2 = self.split_set[idx]['img2']
        path_disp1 = self.split_set[idx]['disp1']
        path_calib = self.split_set[idx]['calib']

        # 载入双目图像
        img1 = self.load_image(path_img1)       # 读取双目图像
        img2 = self.load_image(path_img2)
        disp1 = self.load_disp(path_disp1)      # 读取视差图
        calib = self.load_calib(path_calib)

        try:
            path_ncc1 = self.split_set[idx]['mask1']
            mask1 = self.load_mask(path_ncc1)       # 读取非遮挡区域掩码
        except:
            mask1 = Image.fromarray(np.zeros_like(np.array(disp1)))

        # 读取内外参 (基线)
        baseline = calib['baseline']
        K1, K2 = calib['cam0'], calib['cam1']

        # 深度计算
        dep1 = (K1[0, 2] - K2[0, 2]) + K1[0, 0] * baseline / (np.array(disp1) + 1e-8)
        dep1 = dep1 * (np.array(disp1) > 1e-3).astype(dep1.dtype)
        dep1 = Image.fromarray(dep1)

        return {"img_left": img1, "img_right": img2, 
                "gt_disp_left":disp1, "gt_disp_right": None,
                "gt_dep_left": dep1, "gt_dep_right": None,
                "mask_left": mask1, "mask_right":None,
                "K1": torch.from_numpy(K1), "K2": torch.from_numpy(K2), "b":baseline,
                "ref_path": path_img1}

    def __getitem__(self, idx):

        one_sample = self._load_data(idx)       # 加载数据
        ori_sample = copy.deepcopy(one_sample)  # save the origin sample
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

        # mask1
        mask1 = TF.to_tensor(np.array(one_sample['mask_left']))

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