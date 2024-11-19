"""
Data distributed parallel utils. Copied from DINO.
"""
import os
import sys
import time
import math
import random
import datetime
import subprocess
from collections import defaultdict, deque

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from PIL import ImageFilter, ImageOps


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def get_all_gather(tensor):
    if not is_dist_avail_and_initialized():
        print("Not in distributed mode.")
        return tensor
    tensor_list = [tensor.clone() for _ in range(get_world_size())]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode():
    # launched with `torch.distributed.launch`
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # 当前处于某个进程中, 配置当前进程的环境参数
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    # launched naively with `python main.py`
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        rank, gpu, world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'     # 针对单机单卡的情况设置通信地址和端口
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",             # 后端通信采用 NVIDIA 的推荐协议
        init_method='env://',       # 使用环境变量初始化 (MASTER_PORT, MASTER_ADDR)
        world_size=world_size,      # 总进程数
        rank=rank,                  # 当前进程序号
    )

    torch.cuda.set_device(gpu)      # 设置当前节点 local_rank 对应的 GPU id
    print('| distributed init (rank {}): {}'.format(rank, "env://"), flush=True)
    dist.barrier()                  # 同步屏障, 所有节点启动后才进入下一句
    setup_for_distributed(rank == 0)
    print("MASTER ADDR, PORT: {}:{}".format(os.environ['MASTER_ADDR'], os.environ['MASTER_PORT']))
