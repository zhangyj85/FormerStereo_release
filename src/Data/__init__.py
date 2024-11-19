"""
description: data loader
last modify: 2023.03.20
author: Yongjian Zhang
"""

import os
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset
from .sceneflow_dataset import SceneFlowDataset
from .crestereo_dataset import CREStereoDataset
from .kitti2015 import KITTIStereo2015
from .kitti2012 import KITTIStereo2012
from .middlebury import MiddleburyDataset
from .eth3d_dataset import ETH3DDataset
# from .drivingstereo import DrivingStereo
# from .oxford_dataset import OxfordDataset


__datasets__ = {
    "sceneflow": SceneFlowDataset,
    "kitti2015": KITTIStereo2015,
    "kitti2012": KITTIStereo2012,
    "crestereo": CREStereoDataset,
    "middlebury": MiddleburyDataset,
    "eth3d": ETH3DDataset,
    # "drivingstereo": DrivingStereo,
    # "oxford": OxfordDataset,
}


def get_loader(config):

    # 环境配置
    cfg_mode = config['mode'].lower()
    pin_memory = True   # 内存充足, 设置pin_memory=True可加快tensor到GPU进程; 当系统卡住，或交换内存使用过多，设置pin_memory=False
    num_workers = 4     # 通常来说, 4 * GPUs 会比较好

    rvc_datasets1 = ['sceneflow'] + ['middlebury'] * 13 + ['eth3d'] * 73         # stage 1, 90% SF + 5% MD + 5% ETH
    rvc_datasets2 = ['sceneflow'] + ['middlebury'] * 29 + ['eth3d'] * 164 + ['kitti2012'] * 228 + ['kitti2015'] * 222
    rvc_datasets3 = ['sceneflow'] + ['crestereo'] + ['middlebury'] * 86 + ['eth3d'] * 485  # stage 1, 90% (SF+CRE) + 5% MD + 5% ETH
    rvc_datasets4 = ['sceneflow'] + ['crestereo'] + ['middlebury'] * 600 + ['eth3d'] * 600 + ['kitti2015'] * 580 + ['kitti2012'] * 580 # stage 1, 40% (SF+CRE) + 5% MD + 5% ETH + 50% KT
    manual_dataset = config['data']['datasets']
    use_for_training_list = rvc_datasets4

    # 训练数据集
    if cfg_mode == 'train':
        loaders_list = []
        for data_name in use_for_training_list:
            data_name = data_name.lower()
            data_class = __datasets__[data_name]
            config['data']['json_file'] = os.path.join(config['data']['data_json'], data_name + ".json")
            data_loader = data_class(config, mode='train', split='train')
            loaders_list.append(data_loader)
        datasets = ConcatDataset(loaders_list)
        sampler = DistributedSampler(datasets, shuffle=True)
        train_loader = DataLoader(
            dataset=datasets,
            sampler=sampler,
            batch_size=int(config['train']['batch_size'] * config['train']['accumulate_grad_iters']),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        return train_loader

    # 验证 or 测试数据集
    elif cfg_mode == 'test':
        # 测试阶段仅使用单进程, 且不需要 shuffle, 无需写入 sampler
        data_name = config['data']['datasets'][0].lower()
        data_class = __datasets__[data_name]
        config['data']['json_file'] = os.path.join(config['data']['data_json'], data_name + ".json")
        dataset = data_class(config, mode='test', split=config['split'].lower())
        test_loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=pin_memory,
            drop_last=False
        )
        return test_loader
    else:
        raise NotImplementedError('Mode [{:s}] is not supported.'.format(cfg_mode))
