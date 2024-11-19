"""
Description:
    Project: XYZ Robotics
    Function: test and save(optional) the model performance.
    Author: Yongjian Zhang
    E-mail: zhangyj85@mail2.sysu.edu.cn
"""
import sys
import torch
import os
import time
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from Data import get_loader
from Models import MODEL
from Metrics import METRICS
import imageio
import skimage

from utils.tools import *
from utils.logger import ColorLogger
from utils.visualization import disp_error_image_func, disp_color_func

# 加载进度条, 可视化训练进度
from tqdm import tqdm


def save_pfm(file, image, scale = 1):
  color = None

  if image.dtype.name != 'float32':
    raise Exception('Image dtype must be float32.')

  if len(image.shape) == 3 and image.shape[2] == 3: # color image
    color = True
  elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
    color = False
  else:
    raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

  file.write('PF\n' if color else 'Pf\n')
  file.write('%d %d\n' % (image.shape[1], image.shape[0]))

  endian = image.dtype.byteorder

  if endian == '<' or endian == '=' and sys.byteorder == 'little':
    scale = -scale

  file.write('%f\n' % scale)

  image.tofile(file)


def submit_kitti(pred, ttime, ref_path, rootpath):
    scene_name = os.path.basename(ref_path)[:-4]
    pred_dir = os.path.join(rootpath, 'disp_0')
    os.makedirs(pred_dir, exist_ok=True)
    pred_path = os.path.join(pred_dir, scene_name + '.png')
    disp_est_uint = np.round(pred * 256).astype(np.uint16)
    skimage.io.imsave(pred_path, disp_est_uint)


def submit_eth3d(pred, ttime, ref_path, rootpath):
    scene_name = os.path.basename(os.path.dirname(ref_path))
    pred_dir = os.path.join(rootpath, 'low_res_two_view')
    os.makedirs(pred_dir, exist_ok=True)
    pred_path = os.path.join(pred_dir, scene_name + '.pfm')
    time_path = os.path.join(pred_dir, scene_name + '.txt')
    with open(pred_path, 'w') as f:
        save_pfm(f, pred[::-1, :])
    with open(time_path, 'w') as timing_file:
        timing_file.write('runtime ' + str(ttime))


def submit_middlebury(pred, ttime, ref_path, rootpath):
    scene_name = os.path.basename(os.path.dirname(ref_path))
    split = "testF"     # ["trainingF", "trainingH", "testF"]
    pred_dir = os.path.join(rootpath, split, scene_name)
    os.makedirs(pred_dir, exist_ok=True)
    pred_path = os.path.join(pred_dir, 'disp0FormerRaft_RVC.pfm')
    time_path = os.path.join(pred_dir, 'timeFormerRaft_RVC.txt')
    with open(pred_path, 'w') as f:
        save_pfm(f, pred[::-1, :])
    with open(time_path, 'w') as timing_file:
        timing_file.write(str(ttime))


__submit_func__ = {
    "kitti2015": submit_kitti,
    "kitti2012": submit_kitti,
    "eth3d": submit_eth3d,
    "middlebury": submit_middlebury,
}


class SubmitSolver(object):
    def __init__(self, config):
        self.config = config
        self.savepath = os.path.join(config['record']['path'], config['data']['datasets'][0], config['split'])          # if save results, save in savepath
        
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

    def load_checkpoint(self):
        ckpt_full = self.config['train']['resume']
        states = torch.load(ckpt_full, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(states['model_state'], strict=True)     # 忽略不匹配的 module

    def save_results(self, imgL, imgR, disp_pt, disp_gt, save_path):

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
        color_pt_disp = color_pt_disp.float() * ((disp_pt > 0.).float()).permute(0, 2, 3, 1)  # 将无效深度置为 0
        color_pt_disp = color_pt_disp.type(torch.uint8)
        self.Imagetool.ImageSave(color_pt_disp[0], save_path + "/left_color_disp_pt.png")

        # save disparity error map
        color_error = disp_error_image_func(disp_pt.squeeze(1), disp_gt.squeeze(1)).to(disp_gt.device)  # B C H W
        color_error = (color_error.float() * 255 * (disp_gt > 0.1).float()).permute(0, 2, 3, 1)                 # B H W C
        color_error = color_error.type(torch.uint8).data.cpu().numpy()
        imageio.imwrite(save_path + "/disp_err.png", color_error[0])

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

            metrics_dict = {'time': [], 'density': []}
            N_total, idx = 0, 0
            for test_batch in tqdm(self.test_loader):
                imgL   = test_batch["ir1"].to('cuda', non_blocking=True)
                imgR   = test_batch["ir2"].to('cuda', non_blocking=True)
                disp_L = test_batch["gt1"].to('cuda', non_blocking=True)
                if len(test_batch["mask1"]) > 0:
                    ncc_mask_L = test_batch["mask1"].to('cuda', non_blocking=True)
                    ncc_mask_L[ncc_mask_L < 0.999] = 0  # 参考 GraftNet & ITSA, 仅保留 I=255 的真值
                else:
                    ncc_mask_L = None

                N_curr = imgL.shape[0]
                assert N_curr == 1, "Only support one sample during inference."

                # 准确计算程序运行时间, 需要考虑 cuda 异步计算的最大时长
                torch.cuda.synchronize()
                start_time = time.time()

                if self.config['data']['datasets'][0]=='middlebury' and self.config['split'] in ['train-f', 'test-f']:
                    _, _, h, w = imgL.shape
                    imgL_ = F.interpolate(imgL, size=(1120, 1568), mode="bilinear", align_corners=True)
                    imgR_ = F.interpolate(imgR, size=(1120, 1568), mode="bilinear", align_corners=True)
                    output = self.model(imgL_, imgR_)
                    disp_pred = F.interpolate(output["disparity"] * w / 1568, size=(h, w), mode="bilinear", align_corners=True)
                    disp_pred = disp_pred.clamp(self.min_disp, self.max_disp)   # 输出结果限制在有效范围内
                else:
                    output = self.model(imgL, imgR)
                    disp_pred = output["disparity"].clamp(self.min_disp, self.max_disp)  # 输出结果限制在有效范围内

                torch.cuda.synchronize()
                infer_time = time.time() - start_time

                if self.config['record']['submit']:
                    disp_pred_np = disp_pred.data.cpu().numpy()
                    disp_pred_np = disp_pred_np[0, 0].astype(np.float32)
                    submit_dataset = self.config['data']['datasets'][0]
                    ref_file_path = test_batch["ref_path"][0]
                    __submit_func__[submit_dataset](disp_pred_np, infer_time, ref_file_path, self.savepath)

                # Mask 指标, 预测结果的稠密程度
                metrics_dict['density'].append(1.0 * N_curr)

                # 视差估计效果
                eval_results = self.eval_metrics(disp_L, disp_pred, training=False, ncc_mask=ncc_mask_L)
                for key, val in eval_results.items():
                    if key not in metrics_dict.keys():
                        metrics_dict[key] = []  # initial key list
                    metrics_dict[key].append(val * N_curr)

                # 时间指标
                metrics_dict['time'].append(infer_time / N_curr)    # 时间应当根据 batch size 进行平均

                # save results?
                if self.config['record']['vis']:
                    subpath = os.path.join(self.savepath, 'vis', '{:06d}'.format(idx))
                    os.makedirs(subpath, exist_ok=True)
                    self.save_results(imgL, imgR, disp_pred, disp_L, subpath)
                    idx += 1

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
