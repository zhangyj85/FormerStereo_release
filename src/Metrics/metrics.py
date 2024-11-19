import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_occlusion_mask(disparity):
    # disparity in left view
    _, _, _, W = disparity.shape
    index = torch.arange(0, W, device=disparity.device, requires_grad=False).view(1, 1, 1, W)
    matching_right = index - disparity      # W_r = W_l - Disp
    # invisible region
    visible = (matching_right > 0.).float() # B1HW
    # occlusion region
    count = 0.
    for i in range(1, W):
        shift_map = F.pad(matching_right[:, :, :, i:], (0, i, 0, 0), mode='constant', value=-1)     # shift in left
        count = count + (torch.abs(matching_right - shift_map) < 0.5).float()                       # 0.5 means round
    occlud = (count > 0.).float()
    # TODO: 增加形态学处理去除孔洞
    # 最终得到的不包含invisible和occlusion的mask
    valid_mask = visible * (1 - occlud)
    return valid_mask.bool().float().detach()


def get_edge_mask(disparity, thresh=10., dilation=10):
    # disparity in left view, similarity to SMD-Nets
    def gradient_x(img):
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]
        return gx

    def gradient_y(img):
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gy

    # 一阶梯度
    gx_disp = torch.abs(gradient_x(disparity))
    gy_disp = torch.abs(gradient_y(disparity))

    # 得到边缘
    edges = (gx_disp > thresh).float() + (gy_disp > thresh).float()
    edges = (edges > 0.).float()

    # 对边缘进行膨胀, 以考虑边缘附近的像素估计精度
    if dilation > 0:
        edge_list = []
        kernel = np.ones((dilation, dilation), np.uint8)
        for i in range(disparity.shape[0]):
            edge_slice = cv2.dilate(edges[i,0,...].data.cpu().numpy(), kernel, iterations=1)
            edge_list.append(edge_slice)
        edges = np.stack(edge_list, axis=0)         # HW -> BHW
        edges = torch.from_numpy(edges)             # To Tensor
        edges = torch.unsqueeze(edges, dim=1)       # B1HW
        edges = edges.to(disparity.device)          # change device

    return edges


def IoUpoint(target, output, **args):
    """
    input args:
    target: G.T. unocclusion mask, 0-1 float
    output: P.T. unocclusion mask, 0-1 float

    output args:
    ExtraPoint:   实际被遮挡, 但是计算出了深度值的点, 用于分析边缘生长出多少. 采用 IoU 计算方法;
    InvalidPoint: 未被遮挡, 但是没算出深度值的点;
    """
    # print((((1 - target) * output) > 1e-3).sum())
    # print(((1 - target) > 1e-3).sum())
    ExtraPoint = (((1 - target) * output) > 1e-3).float().sum() / ((1 - target) > 1e-3).float().sum()       # (GT_occlu ∩ output_valid) / GT_occlu
    InvalidPoint = ((target * (1 - output)) > 1e-3).float().sum() / (target > 1e-3).float().sum()             # 1 - (GT_valid ∩ output_valid) / GT_valid
    return ExtraPoint, InvalidPoint


def epe_metric(target, output, mask):
    """
        target: G.T. disparity or depth, tensor, B1HW
        output: P.T. disparity or depth, tensor, B1HW
        valid:  A mask, which region use for calculate error, B1HW, bool
    """
    target, output = target[mask], output[mask]                 # 考虑有效区域
    err = torch.abs(target - output)                            # L1 误差
    avg = torch.mean(err)                                       # 误差均值
    return avg.data.cpu()


def d1_metric(target, output, mask):
    target, output = target[mask], output[mask]                 # 考虑有效区域
    err = torch.abs(target - output)                            # L1 误差
    err_mask = (err > 3) & (err / target > 0.05)                # 超过3像素, 且相对估计误差大于5%, 视为 D1 误差
    err_mean = torch.mean(err_mask.float())                     # (0,1), 百分比
    return err_mean.data.cpu()


def thresh_metric(target, output, mask, threshold=3.0):
    target, output = target[mask], output[mask]                 # 考虑有效区域
    err = torch.abs(target - output)                            # L1 误差
    err_mask = (err > threshold)                                # 超出阈值
    err_mean = torch.mean(err_mask.float())                     # (0,1), 百分比
    return err_mean.data.cpu()


class METRICS(nn.Module):
    def __init__(self, config):
        super(METRICS, self).__init__()
        """Metrics in different benchmark
        Synthetics Datasets:
        
        Real-world Datasets:
        # KITTI 2012: δ_x (x=3), EPE. Both including occ and non-occ
        # KITTI 2015: δ_X. Including foreground, background, and all-pixels
        # Middlebury 2014: δ_x (x=0.5, 1, 2, 4), EPE, rms. The δ_x and EPE including non-occ and all.
        """
        # following ITSA, GraftNet, the max disparity only considering 192
        self.max_disp = 192.        # 所有数据集仅考虑 192 视差以内的结果
        self.min_disp = 0.
        print("The metrics only considers the valid disparity ranging from {:.0f} to {:.0f}.".format(self.min_disp, self.max_disp))

    def forward(self, target, output, training=False, ncc_mask=None):

        # 考虑所有候选视差范围内的结果
        gt_concern = (self.min_disp < target) & (target < self.max_disp)
        gt_concern.detach_()

        # 若预测视差非稠密, 则统计有效部分 [0, +inf]
        pt_concern = (0 < output) & (output < np.inf)
        pt_concern.detach_()

        # init candidate metrics
        metrics = {
            "Bad0.5": 0.,       # err > 0.5px 的像素占有效估计像素的百分比
            "Bad1.0": 0.,
            "Bad2.0": 0.,
            "Bad3.0": 0.,
            "D1-all": 0.,       # kitti 评测指标
            "all-EPE": 0.,      # 有效像素内的平均端点误差
            "ncc-Bad0.5": 0.,
            "ncc-Bad1.0": 0.,
            "ncc-Bad2.0": 0.,
            "ncc-Bad3.0": 0.,
            "D1-ncc": 0.,
            "ncc-EPE": 0.,      # 非遮挡区域内的平均端点误差
            "occ-EPE": 0.,      # 遮挡区域内的平均端点误差
            "edge-EPE": 0.,     # 边缘区域内的平均端点误差
        }

        # 有效像素内的误差
        # TODO: Following GwcNet, remove all the images with less than 10% valid pixels (0≤d<Dmax) in the test set
        concern = gt_concern & pt_concern
        if training:
            if concern.any():
                # 训练期间只需要两个关键参数, 其余不参与计算, 节省计算资源
                metrics['Bad3.0'] = thresh_metric(target, output, concern, threshold=3.0).item()
                metrics['all-EPE'] = epe_metric(target, output, concern).item()
            return metrics

        if concern.any():       # 若存在一个元素为 True
            metrics['Bad0.5'] = thresh_metric(target, output, concern, threshold=0.5).item()
            metrics['Bad1.0'] = thresh_metric(target, output, concern, threshold=1.0).item()
            metrics['Bad2.0'] = thresh_metric(target, output, concern, threshold=2.0).item()
            metrics['Bad3.0'] = thresh_metric(target, output, concern, threshold=3.0).item()
            metrics['D1-all'] = d1_metric(target, output, concern).item()
            metrics['all-EPE'] = epe_metric(target, output, concern).item()

        # 遮挡/ 非遮挡区域内的平均端点误差
        if ncc_mask is None:
            # 没有真值遮挡图, 则根据真值生成(需要视差真值密度为100%, 结果才可信)
            ncc_mask = get_occlusion_mask(target)       # 同名点 mask, float
        ncc_concern = ncc_mask.bool() & concern
        if ncc_concern.any():
            metrics['ncc-Bad0.5'] = thresh_metric(target, output, ncc_concern, threshold=0.5).item()
            metrics['ncc-Bad1.0'] = thresh_metric(target, output, ncc_concern, threshold=1.0).item()
            metrics['ncc-Bad2.0'] = thresh_metric(target, output, ncc_concern, threshold=2.0).item()
            metrics['ncc-Bad3.0'] = thresh_metric(target, output, ncc_concern, threshold=3.0).item()
            metrics['D1-ncc'] = d1_metric(target, output, ncc_concern).item()
            metrics['ncc-EPE'] = epe_metric(target, output, ncc_concern).item()
        occ_mask = 1 - ncc_mask
        occ_concern = occ_mask.bool() & concern
        if occ_concern.any():
            metrics['occ-EPE'] = epe_metric(target, output, occ_concern).item()

        # 边缘区域内的平均端点误差
        # TODO: following SMD-Nets to use soft edge error
        edge_mask = get_edge_mask(disparity=target, thresh=1., dilation=5)
        edge_concern = edge_mask.bool() & concern
        if edge_concern.any(): metrics['edge-EPE'] = epe_metric(target, output, edge_concern).item()

        return metrics
