import os
import torch
import argparse
import random
import json
from tqdm import tqdm

import sys
sys.path.append("..")
from Options import parse_opt
from Data import get_loader

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--options', type=str, help='Path to the option JSON file.', default='../Options/options.json')
args = parser.parse_args()
args = parse_opt(args.options)

# 固定随机数种子, 确保每次生成的地址文件在顺序上是固定的
random.seed(args["environment"]["seed"])


def analyse_disparity_distribution(dataset, max_disp, min_disp, num_split, epoch_num=1):

    # 数据预定义
    step = (max_disp - min_disp) // num_split
    num_split = (max_disp - min_disp) // step   # 更新分类总数
    splits = [x for x in range(min_disp, max_disp+step, step)]     # num_split+1 个视差区间切片
    splits = torch.tensor(splits, device='cuda').view(1, -1, 1, 1)      # (B,D,H,W)

    batch_list = []  # 迭代计数器
    ratio = torch.zeros(1, num_split, device='cuda')

    # 对数据进行处理
    with torch.no_grad():
        for epoch in range(epoch_num):
            # 实验证明, 多次迭代 epoch 并不会改变数据集本身的视差分布属性, 输入的图像本来就是从原分布均匀采样
            for batch in tqdm(dataset):
                disp = batch["gt1"].to('cuda', non_blocking=True)   # (B,1,H,W)
                B = disp.shape[0]
                disp = disp.repeat(1, num_split, 1, 1)              # (B,D,H,W)
                ceil = (splits[:,1:,:,:] - disp) >= 0                          # 后 d-1 个有效
                floor = (disp - splits[:,:-1,:,:]) > 0                         # 前 d-1 个有效, 不考虑0, 左开右闭
                # 区间划分, 大于 floor 为真且小于 ceil 为真
                devided_disp_map = torch.logical_and(floor, ceil)
                # 区间数量统计, 采用累加原则
                devided_disp_map = devided_disp_map.float().view(B, num_split, -1)
                devided_disp_map = torch.mean(devided_disp_map, dim=-1, keepdim=False)  # (B,D), 沿空间维度求平均
                ratio = sum(batch_list) * ratio + torch.sum(devided_disp_map, dim=0, keepdim=True)  # (1,D), 沿batch维度求和
                batch_list.append(B)
                ratio = ratio / sum(batch_list)
    return ratio


if __name__ == "__main__":
    # 数据加载
    args['data']['data_json'] = os.path.join("..", args['data']['data_json'])
    if args['mode'] == 'train':
        dataset = get_loader(args)[0]
        epoch_num = args['train']['epoch']
    else:
        dataset = get_loader(args)
        epoch_num = 1

    max_disp = args['model']['max_disp']
    min_disp = args['model']['min_disp']
    num_split = max_disp - min_disp
    disparity_distribution = analyse_disparity_distribution(
        dataset, max_disp=max_disp, min_disp=min_disp, num_split=num_split, epoch_num=epoch_num
    )
    print("")
    plt.figure(), plt.plot(disparity_distribution[0].data.cpu())
    plt.show()
