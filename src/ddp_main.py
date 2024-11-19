import argparse
import torch
import numpy as np
from torch.backends import cudnn
from Options import parse_opt
from utils.distributed_utils import init_distributed_mode, get_rank


def main():
    # Parse arguments.
    parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--options', type=str, help='Path to the option JSON file.', default='./Options/options.json')
    args = parser.parse_args()

    # 全局同一环境配置
    opt = parse_opt(args.options)
    init_distributed_mode()     # 分布式进程初始化

    # Deterministic Settings.
    if opt['environment']['deterministic']:
        # fix the seed for reproducibility
        seed = opt['environment']['seed'] + get_rank() + 1
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True      # 固定每次返回的卷积算法结果
        cudnn.benchmark = True          # cudnn 加速
    else:
        cudnn.deterministic = False     # 不限制随机数种子, 算法存在随机性
        cudnn.benchmark = True
    
    # Create solver.
    from Sovlers import get_solver
    solver = get_solver(opt)

    # Run.
    solver.run()


if __name__ == "__main__":
    main()
