"""
loading files of oxford stereo data
Author: Yongjian Zhang
E-mail: zhangyj85@mail2.sysu.edu.cn
Date:   2023.11.14
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


from Data.oxford_sdk.build_pointcloud import build_pointcloud
from Data.oxford_sdk.transform import build_se3_transform
from Data.oxford_sdk.image import load_image
from Data.oxford_sdk.camera_model import CameraModel


class OxfordDataset(Dataset):
    def __init__(self, args, mode, split='train'):
        super(OxfordDataset, self).__init__()
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
                                augmentation.Resize(self.args['crop_size']),
                                augmentation.VerticalFlip(),
                                # augmentation.RandomVdisp(),
                                augmentation.RandomCrop(self.args['crop_size'], shift=False,
                                                        max_disp=args["model"]["max_disp"],
                                                        min_disp=args["model"]["min_disp"]),
                                ### photometric augmentation
                                augmentation.ColorJitters(asymmetric=True),
                                augmentation.ColorConvert(convert_left=False),
                                # augmentation.Reflection(self.args['crop_size'], max_disp=args["model"]["max_disp"]),
                                # augmentation.RandomOcclusion()
                                ])
        else:
            self.augmentation = augmentation.Compose([augmentation.RandomCrop(self.args['crop_size'], shift=False)])

    def __len__(self):
        return len(self.split_set)

    def _load_data(self, idx):

        # 路径导入
        path_model_dir = "/media/zhangyj85/Dataset/Stereo_Datasets/Oxford/Downloads/camera_models/"
        path_extri_dir = "/media/zhangyj85/Dataset/Stereo_Datasets/Oxford/robotcar-dataset-sdk-3.1/extrinsics/"
        path_laser_dir = self.split_set[idx]['ldmr']        # 雷达点文件夹
        path_front_dir = self.split_set[idx]['front']
        path_rear_dir = self.split_set[idx]['rear']
        path_img1 = self.split_set[idx]['img1']             # 输入图像
        path_img2 = self.split_set[idx]['img2']
        path_pose = self.split_set[idx]['pose']             # 位姿

        # 载入相机模型
        model = CameraModel(path_model_dir, os.path.dirname(path_img1))
        # 加载图像, 得到图像平面
        image = load_image(path_img1, model)
        # 读取对应的外参
        extrinsics_path = os.path.join(path_extri_dir, model.camera + '.txt')
        with open(extrinsics_path) as extrinsics_file:
            extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]
        # 相机标定
        G_camera_vehicle = build_se3_transform(extrinsics)
        G_camera_posesource = None

        # 读取位姿
        poses_type = re.search('(vo|ins|rtk)\.csv', path_pose).group(1)
        if poses_type in ['ins', 'rtk']:
            with open(os.path.join(path_extri_dir, 'ins.txt')) as extrinsics_file:
                extrinsics = next(extrinsics_file)
                G_camera_posesource = G_camera_vehicle * build_se3_transform([float(x) for x in extrinsics.split(' ')])
        else:
            # VO frame and vehicle frame are the same
            G_camera_posesource = G_camera_vehicle

        # 读取双目图像的时间戳
        timestamp = int(os.path.basename(path_img1)[:-4])
        # 从点云序列中得到时间戳一定范围内的点云
        pointcloud, reflectance = build_pointcloud(path_laser_dir, path_pose, path_extri_dir,
                                                   timestamp - 1e7, timestamp + 1e7, timestamp)
        pointcloud = np.dot(G_camera_posesource, pointcloud)
        # # front point cloud, 2d lidar没有深度?
        # front_pointcloud, _ = build_pointcloud(path_front_dir, path_pose, path_extri_dir, timestamp - 1e7, timestamp + 1e7, timestamp)
        # front_pointcloud = np.dot(G_camera_posesource, front_pointcloud)
        # # rear point cloud
        # rear_pointcloud, _ = build_pointcloud(path_rear_dir, path_pose, path_extri_dir, timestamp - 1e7, timestamp + 1e7, timestamp)
        # rear_pointcloud = np.dot(G_camera_posesource, rear_pointcloud)
        #
        # all_pointcloud = np.concatenate((pointcloud, front_pointcloud, rear_pointcloud), axis=1)

        # 将点云投影到图像平面
        uv, depth = model.project(pointcloud, image.shape)
        dep = np.zeros(image.shape[:-1])    # image 是 3通道
        dep[np.ravel(uv[1, :]).astype(np.int16), np.ravel(uv[0, :]).astype(np.int16)] = depth   # 将深度填充到图像平面

        # 深度转视差
        focal_y, focal_x = model.focal_length
        center_y, center_x = model.principal_point
        internal = np.array([[focal_x, 0.0, center_x],
                             [0.0, focal_y, center_y],
                             [0.0,     0.0, 1.0]])
        baseline = 0.24     # 单位: m. 长基线 24cm, 短基线 12cm
        disp = focal_x * baseline / (dep + 1e-8) * (dep > 1e-3).astype(dep.dtype)

        # 获得最终结果, 注意对图像车头部分进行裁切
        cut_end = int(image.shape[0] * 0.8)
        img1 = Image.fromarray(image[:cut_end, :, :]).convert("RGB")
        dep1 = Image.fromarray(dep[:cut_end, :])
        disp1 = Image.fromarray(disp[:cut_end, :])

        # 读取右图
        model2 = CameraModel(path_model_dir, os.path.dirname(path_img2))
        img2 = load_image(path_img2, model2)
        img2 = Image.fromarray(img2[:cut_end, :]).convert("RGB")

        # 相机内参
        K1, K2 = internal.copy(), internal.copy()

        return {"img_left": img1, "img_right": img2, 
                "gt_disp_left":disp1, "gt_disp_right": None,
                "gt_dep_left": dep1, "gt_dep_right": None,
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
