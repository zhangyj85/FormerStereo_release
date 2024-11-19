"""
Description:
    Project: XYZ Robotics
    Function: test and save(optional) the model performance.
    Author: Yongjian Zhang
    E-mail: zhangyj85@mail2.sysu.edu.cn
"""

import torch
import os
import time
import cv2
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
from collections import OrderedDict

from Data import get_loader
from Models import MODEL
from Metrics import METRICS
import pdb
import imageio

from utils.tools import *
from utils.logger import ColorLogger
from utils.visualization import disp_error_image_func, disp_color_func

# 加载进度条, 可视化训练进度
from tqdm import tqdm

# occlusion prediction post-process
# from skimage import morphology


class TestSolver(object):
    def __init__(self, config):
        self.config = config
        self.record = config['record']['color'] | config['record']['depth'] | config['record']['point']
        self.savepath = config['record']['path']          # if save results, save in savepath
        
        self.max_disp = self.config['model']['max_disp']
        self.min_disp = self.config['model']['min_disp']

        log_path = os.path.join('../logger', self.config['model']['name'])
        self.logger = ColorLogger(log_path, 'logger.log')

        # 获取模型 & 数据 & 评价指标
        self.model = MODEL(self.config)
        self.test_loader = get_loader(self.config)
        self.eval_metrics = METRICS(self.config)

        # 创建工具包
        image_mean = self.config["data"]["mean"]                                # RGB图, 三通道的归一化数值
        image_std  = self.config["data"]["std"]
        if len(image_std) < 3:
            image_mean, image_std = image_mean * 3, image_std * 3               # 灰度图, 三通道的归一化数值相同
        self.Imagetool = TensorImageTool(mean=image_mean, std=image_std)
        self.PCDTool = Dep2PcdTool()

    def load_checkpoint(self):
        ckpt_full = self.config['train']['resume']
        states = torch.load(ckpt_full, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(states['model_state'], strict=True)     # 忽略不匹配的 module

        # 若存在 teacher module, 则使用 teacher model 的参数替换 student model 的权重
        teacher_ckpt = os.path.join(os.path.dirname(ckpt_full), "teacher_" + os.path.basename(ckpt_full))
        if os.path.isfile(teacher_ckpt):
            self.logger.info("Using teacher params.")
            # 载入 teacher 模型
            teacher_state = torch.load(teacher_ckpt, map_location=lambda storage, loc: storage)
            # 更新字典参数
            for key in states['model_state'].keys():
                subkey = 'module.' + key[16:]
                if subkey in teacher_state['teacher_module'].keys():
                    states['model_state'][key] = teacher_state['teacher_module'][subkey]
            # 载入新的参数
            self.model.load_state_dict(states['model_state'], strict=True)

    def save_results(self, data_in, data_out, save_path):
        # collecting data
        imgL, imgR = data_in["ir1"], data_in["ir2"]
        disp_gt, dep_gt = data_in["gt_disp1"], data_in["gt_dep1"]
        disp_pt, dep_pt = data_out["pt_disp1"], data_out["pt_dep1"]

        if self.config['record']['color']:
            # save color images
            color_imgL = self.Imagetool.renormalize(imgL, fprint=False)
            self.Imagetool.ImageSave(color_imgL[0], save_path + '/left.png')
            color_imgR = self.Imagetool.renormalize(imgR, fprint=False)
            self.Imagetool.ImageSave(color_imgR[0], save_path + '/right.png')

            current_gt_disp = disp_gt[0]
            # 当前视差图的最大视差和最小视差
            valid_mask = (current_gt_disp > 1e-3) & (current_gt_disp < np.inf)                  # 有效范围 [0, +inf]
            if valid_mask.any():
                max_color_disp = np.percentile(current_gt_disp[valid_mask].data.cpu().numpy(), 97)  # 设置可视化的显示范围
                min_color_disp = np.percentile(current_gt_disp[valid_mask].data.cpu().numpy(), 3)
            else:
                max_color_disp = self.max_disp
                min_color_disp = self.min_disp
            # 考虑范围内的最大视差和最小视差
            max_color_disp = min(self.max_disp, max_color_disp)
            min_color_disp = 0.#max(self.min_disp, min_color_disp)

            # save color ground-truth disparity
            color_gt_disp = self.Imagetool.colorize_disp(disp_gt, max_color_disp, min_color_disp, fprint=False)
            color_gt_disp = color_gt_disp.float() * ((disp_gt > 0.1) & (disp_gt < np.inf)).float().permute(0, 2, 3, 1)  # 将无效深度置为 0
            color_gt_disp = color_gt_disp.type(torch.uint8)
            self.Imagetool.ImageSave(color_gt_disp[0], save_path + "/left_color_disp_gt.png")

            # save color estimated disparity
            color_pt_disp = self.Imagetool.colorize_disp(disp_pt, max_color_disp, min_color_disp, fprint=False)
            color_pt_disp = color_pt_disp.float() * ((dep_pt > 0.1).float()).permute(0, 2, 3, 1)  # 将无效深度置为 0
            color_pt_disp = color_pt_disp.type(torch.uint8)
            self.Imagetool.ImageSave(color_pt_disp[0], save_path + "/left_color_disp_pt.png")

            # # save color ground-truth depth
            # current_gt_dep = dep_gt[0]
            # max_color_dep = np.percentile(current_gt_dep[current_gt_dep > 1e-3].data.cpu().numpy(), 97)  # 设置可视化的显示范围
            # min_color_dep = np.percentile(current_gt_dep[current_gt_dep > 1e-3].data.cpu().numpy(), 3)
            # color_gt_dep = self.Imagetool.colorize(dep_gt, max_color_dep, min_color_dep, fprint=False)
            # color_gt_dep = color_gt_dep.float() * (dep_gt > 0.1).float().permute(0, 2, 3, 1)  # 将无效深度置为 0
            # color_gt_dep = color_gt_dep.type(torch.uint8)
            # self.Imagetool.ImageSave(color_gt_dep[0], save_path + "/dep_gt.png")
            #
            # # save estimated depth
            # color_pt_dep = self.Imagetool.colorize(dep_pt, max_color_dep, min_color_dep, fprint=False)
            # color_pt_dep = color_pt_dep.float() * ((dep_pt > 0.1).float()).permute(0, 2, 3, 1)  # 将无效深度置为 0
            # color_pt_dep = color_pt_dep.type(torch.uint8)
            # self.Imagetool.ImageSave(color_pt_dep[0], save_path + "/dep_pt.png")

            # save disparity error map
            color_error = disp_error_image_func(disp_pt.squeeze(1), disp_gt.squeeze(1)).to(disp_gt.device)  # B C H W
            color_error = (color_error.float() * 255 * (disp_gt > 0.1).float()).permute(0, 2, 3, 1)                 # B H W C
            color_error = color_error.type(torch.uint8).data.cpu().numpy()
            imageio.imwrite(save_path + "/disp_err.png", color_error[0])

        if self.config['record']['depth']:
            # 保存 16bit 视差 / 深度 png
            for i in range(disp_pt.shape[0]):
                # 视差为亚像素精度, 精度 0.1 px, 注意scale越小, 则与真实结果的差异越小
                self.Imagetool.DepthSave((disp_pt * (dep_pt > 0.).float())[i].permute(1,2,0), scale=1e-1, save_path=os.path.join(save_path, "%d_disparity_pt.png" % i))
                self.Imagetool.DepthSave((disp_gt * (dep_gt > 0.).float())[i].permute(1,2,0), scale=1e-1, save_path=os.path.join(save_path, "%d_disparity_gt.png" % i))
                # 深度为mm级精度, 精度 0.0001m
                # self.Imagetool.DepthSave(dep_pt[i].permute(1,2,0), scale=1e-4, save_path=os.path.join(save_path, "{%d}_depth_pt.png" % i))
                # self.Imagetool.DepthSave(dep_gt[i].permute(1,2,0), scale=1e-4, save_path=os.path.join(save_path, "{%d}_depth_gt.png" % i))

        if self.config['record']["point"]:
            # 绘制预测点云
            fx, fy, cx, cy = data_in["K1"][0, 0, 0], data_in["K1"][0, 1, 1], \
                             data_in["K1"][0, 0, 2], data_in["K1"][0, 1, 2]
            intrinsic = np.array([fx, 0.0, cx,
                                  0.0, fy, cy,
                                  0.0, 0.0, 1.0]).reshape(3, 3)

            # 预测深度对应的点云
            self.PCDTool.pcd2ply(color_imgL[0].data.cpu().numpy(), dep_pt[0].permute(1, 2, 0).data.cpu().numpy(), intrinsic, save_path + '/depth_pt.ply')
            # 真实深度对应的点云
            self.PCDTool.pcd2ply(color_imgL[0].data.cpu().numpy(), dep_gt[0].permute(1, 2, 0).data.cpu().numpy(), intrinsic, save_path + '/depth_gt.ply')

    def run(self):
        self.model = nn.DataParallel(self.model)
        self.model.cuda()

        self.logger.info("{} Model Testing {}".format("*"*20, "*"*20))     # 输出表头

        if self.config["train"]["resume"] is not None:
            # 若提供了预训练模型, 则加载预训练权重
            self.load_checkpoint()
            self.logger.info('Model loaded: {}, checkpoint: {}.'.format(self.config["model"]["name"], self.config["train"]["resume"]))
        else:
            print("No Model loaded.")

        self.model.eval()

        with torch.no_grad():

            # EPE, Bad0, Bad1, Bad2, Bad3, Edge = [], [], [], [], [], []    # 视差全平面估计平均指标
            # Time = []                                           # 算法推理平均时间
            # Density = []                                        # 预测密度
            # ncc_EPE, occ_EPE = [], []
            metrics_dict = {'time': [], 'density': []}
            N_total, idx = 0, 0

            for test_batch in tqdm(self.test_loader):
                out_dict = {}
                imgL   = test_batch["ir1"].to('cuda', non_blocking=True)
                imgR   = test_batch["ir2"].to('cuda', non_blocking=True)
                disp_L = test_batch["gt1"].to('cuda', non_blocking=True)
                if len(test_batch["mask1"]) > 0:
                    ncc_mask_L = test_batch["mask1"].to('cuda', non_blocking=True)
                    ncc_mask_L[ncc_mask_L < 0.999] = 0  # 参考 GraftNet & ITSA, 仅保留 I=255 的真值
                else:
                    ncc_mask_L = None

                N_curr = imgL.shape[0]

                # 准确计算程序运行时间, 需要考虑 cuda 异步计算的最大时长
                torch.cuda.synchronize()
                start_time = time.time()

                output = self.model(imgL, imgR)
                disp_pred = output["disparity"].clamp(self.min_disp, self.max_disp)   # 输出结果限制在有效范围内

                # 将视差转为深度, 注意采用通用公式
                focal = test_batch["K1"][:, 0, 0, None, None, None].cuda().type_as(disp_pred)
                baseline = test_batch["b"][:, None, None, None].cuda().type_as(disp_pred)
                cx12 = (test_batch["K1"][:,0,2] - test_batch["K2"][:,0,2]).cuda().type_as(disp_pred)
                cx12 = cx12[:, None, None, None]
                dep_pt = focal * baseline / (disp_pred - cx12 + 1e-8) * (disp_pred > 1e-8).float()
                # if self.uncertainty:
                #     mask_pred = output["uncertainty"]
                #     dep_pt = dep_pt * (mask_pred <= self.thresh).float()    # 增加置信度mask

                torch.cuda.synchronize()
                infer_time = time.time() - start_time

                # 计算各个评价指标
                dep_gt = focal * baseline / (disp_L - cx12 + 1e-8) * (disp_L > 1e-8).float()

                #
                out_dict["pt_disp1"] = disp_pred
                out_dict["pt_dep1"] = dep_pt
                test_batch["gt_dep1"] = dep_gt
                test_batch["gt_disp1"] = disp_L


                # Mask 指标, 预测结果的稠密程度
                density = torch.sum(dep_pt > 1e-3).item() / (dep_pt.shape[0] * dep_pt.shape[2] * dep_pt.shape[3])
                metrics_dict['density'].append(density * N_curr)

                # 视差估计效果
                eval_results = self.eval_metrics(disp_L, disp_pred, training=False, ncc_mask=ncc_mask_L)
                for key, val in eval_results.items():
                    if key not in metrics_dict.keys():
                        metrics_dict[key] = []  # initial key list
                    metrics_dict[key].append(val * N_curr)
                # EPE.append(eval_results["all-EPE"] * N_curr)
                # Bad0.append(eval_results["Bad0.5"] * N_curr)
                # Bad1.append(eval_results["Bad1.0"] * N_curr)
                # Bad2.append(eval_results["Bad2.0"] * N_curr)
                # Bad3.append(eval_results["Bad3.0"] * N_curr)
                # Edge.append(eval_results["edge-EPE"] * N_curr)
                # ncc_EPE.append(eval_results["ncc-EPE"] * N_curr)
                # occ_EPE.append(eval_results["occ-EPE"] * N_curr)
                #
                # # if EPE[-1] > 15:
                # #     print("pause")

                # 时间指标
                metrics_dict['time'].append(infer_time / N_curr)    # 时间应当根据 batch size 进行平均

                # save results?
                if self.record: # 记录一下 hard 场景
                    subpath = self.savepath + '/' + self.config["data"]["datasets"][0] + '/' + self.config["split"] + '/' + self.config["model"]["name"] + '/{:08d}'.format(idx)        # 保存到指定文件夹
                    if not os.path.exists(subpath):
                        os.makedirs(subpath)
                    self.save_results(test_batch, out_dict, subpath)
                    idx += 1

                    file_path = self.savepath + '/' + self.config["data"]["datasets"][0] + '/' + self.config[
                        "split"] + '/' + self.config["model"]["name"] + "/res.txt"
                    with open(file_path, "a") as f:
                        f.write(f"{idx: >08d}: {metrics_dict['all-EPE'][-1]: .5f} {100*metrics_dict['Bad2.0'][-1]: .3f}\n")

                N_total += N_curr

        # 记录当前模型的各项指标, 除去第一个, 因为在加速比较耗时
        self.logger.info("Test result in {} frames. Average inferance time is {:3f}s.".format(N_total, sum(metrics_dict['time'][1:]) / (N_total-1)))

        del metrics_dict['time']
        for key, val in metrics_dict.items():
            val = np.array(val)                     # list -> numpy, so that for min, med, max and avg
            str = 'px' if 'EPE' in key else '100%'  # 单位
            self.logger.info(
                '{:12s}: min:{:6f}, med:{:6f}, max:{:6f}, avg:{:6f}'.format(
                    key + "(%s)"%(str), np.min(val), np.median(val), np.max(val), np.sum(val) / N_total
                )
            )
