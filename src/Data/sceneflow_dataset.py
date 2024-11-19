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
import matplotlib.pyplot as plt
import copy
import cv2

# def census_transform(image, window_size=3):
#     """
#     Take a gray scale image and for each pixel around the center of the window generate a bit value of length
#     window_size * 2 - 1. window_size of 3 produces bit length of 8, and 5 produces 24.
#
#     The image gets border of zero padded pixels half the window size.
#
#     Bits are set to one if pixel under consideration is greater than the center, otherwise zero.
#
#     :param image: numpy.ndarray(shape=(MxN), dtype=numpy.uint8)
#     :param window_size: int odd-valued
#     :return: numpy.ndarray(shape=(MxN), , dtype=numpy.uint8)
#     """
#     half_window_size = window_size // 2
#
#     image = cv2.copyMakeBorder(image, top=half_window_size, left=half_window_size, right=half_window_size, bottom=half_window_size, borderType=cv2.BORDER_CONSTANT, value=0)
#     rows, cols = image.shape
#     census = np.zeros((rows - half_window_size * 2, cols - half_window_size * 2), dtype=np.uint8)
#     center_pixels = image[half_window_size:rows - half_window_size, half_window_size:cols - half_window_size]
#
#     offsets = [(row, col) for row in range(half_window_size) for col in range(half_window_size) if not row == half_window_size + 1 == col]
#     for (row, col) in offsets:
#         # 如果邻域大于中心像素, 则左移一位, | 表示按位或
#         census = (census << 1) | (image[row:row + rows - half_window_size * 2, col:col + cols - half_window_size * 2] >= center_pixels)
#     return cv2.normalize(census, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

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


"""
The virtual imaging sensor has a size of 32.0mmx18.0mm.
Most scenes use a virtual focal length of 35.0mm. For those scenes, the virtual camera intrinsics matrix is given by

fx=1050.0   0.0 cx=479.5
0.0 fy=1050.0   cy=269.5
0.0 0.0 1.0
where (fx,fy) are focal lengths and (cx,cy) denotes the principal point.

Some scenes in the Driving subset use a virtual focal length of 15.0mm (the directory structure describes this clearly). 
For those scenes, the intrinsics matrix is given by

fx=450.0    0.0 cx=479.5
0.0 fy=450.0    cy=269.5
0.0 0.0 1.0

"""

class SceneFlowDataset(Dataset):
    def __init__(self, args, mode, split='train'):
        super(SceneFlowDataset, self).__init__()
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
                                augmentation.ColorJitters(asymmetric=True),
                                augmentation.ColorConvert(convert_left=False),
                                # augmentation.Reflection(self.args['crop_size'], max_disp=args["model"]["max_disp"]),
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
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        data = Image.fromarray(data.astype('float32'), mode='F')
        return data

    def load_calib(self, calib_path, frame_idx):
        lines = read_all_lines(calib_path)          # 读取到所有行, 3行1组, [frame idx; L P4*4; R P4*4, []], note: frame idx 并不都是从 0 开始的
        start_idx = lines.index("Frame {}".format(frame_idx))
        calib = lines[start_idx:start_idx+4]
        P_img1 = np.reshape(np.array([float(str) for str in calib[1].split()[1:]]), (4,4))
        P_img2 = np.reshape(np.array([float(str) for str in calib[2].split()[1:]]), (4,4))
        return P_img1, P_img2

    def _load_data(self, idx):
        path_img1 = self.split_set[idx]['img1']
        path_img2 = self.split_set[idx]['img2']
        path_disp1 = self.split_set[idx]['disp1']
        path_disp2 = self.split_set[idx]['disp2']
        path_calib = self.split_set[idx]['calib']

        # 载入双目图像
        img1 = self.load_image(path_img1)       # 读取双目图像
        img2 = self.load_image(path_img2)
        disp1 = self.load_disp(path_disp1)      # 读取视差图
        disp2 = self.load_disp(path_disp2)

        # 读取外参 (基线)
        if path_calib == None:
            baseline = 1.0                      # 近似值
        else:
            [P1, P2] = self.load_calib(path_calib, frame_idx=int(self.split_set[idx]["frame_idx"]))
            baseline = P2[0,3] / P2[0,0] - P1[0,3] / P1[0,0]

        # 读取内参 (焦距)
        if "15mm_focallength" in path_img1:
            internal = np.array([[450.0, 0.0,   479.5],
                                 [0.0,   450.0, 269.5],
                                 [0.0,   0.0,   1.0]])
        else:
            internal = np.array([[1050.0, 0.0,    479.5],
                                 [0.0,    1050.0, 269.5],
                                 [0.0,    0.0,    1.0]])
        K1, K2 = internal.copy(), internal.copy()

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