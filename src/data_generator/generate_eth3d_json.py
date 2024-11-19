"""
    工程项目: XYZ 主动双目深度探测, 生成数据 sample 所需路径
    张勇健， 17308223， zhangyj85@mail2.sysu.edu.cn
    last modify: 2023.03.13
    ======================================================================

    This script generates a json file for CREStereo dataset.
    python utils/generate_crestereo_json.py
"""

import os
import glob
import random
random.seed(0)


def generate_json(dict_json, rootpath="/media/zhangyj85/Dataset/Stereo Datasets/ETH3D", split='train'):
    # 生成包含 left + right + gt 相对路径的.json文件
    # 用于生成训练集

    # For train splits
    dict_json = dict_json
    cnt_seq = 0  # scence计数

    imgs_filepaths = glob.glob(os.path.join(rootpath, "**/im0.png"), recursive=True)
    imgs_filepaths.sort()
    print("Total samples in ETH3D: {}".format(len(imgs_filepaths)))
    for path in imgs_filepaths:
        dir_name = os.path.dirname(path)
        path_img1 = os.path.join(dir_name, 'im0.png')
        path_img2 = os.path.join(dir_name, 'im1.png')
        path_disp = os.path.join(dir_name, 'disp0GT.pfm')
        path_ncc1 = os.path.join(dir_name, 'mask0nocc.png')
        path_calib = os.path.join(dir_name, 'calib.txt')
        dict_sample = {
            "img1": path_img1,
            "img2": path_img2,
            "disp1": path_disp,
            "mask1": path_ncc1,
            "calib": path_calib,
        }

        if 'two_view_testing' in dir_name:
            del dict_sample['disp1'], dict_sample['mask1']
            dict_sample['disp1'] = os.path.join(dir_name.replace('two_view_testing', 'test_ref/mix'), 'disp0pred.pfm')

        # 验证路径字典中的每个元素（value）是否存在
        flag_valid = True
        for key in dict_sample:
            flag_valid &= os.path.exists(dict_sample[key])
            if not flag_valid:
                print("Invalid key: {}, path: {}".format(key, dict_sample[key]))
                break

        if not flag_valid:
            continue

        dict_json[split].append(dict_sample)
        cnt_seq += 1

    print("Add {} samples, Total {} samples in {} set".format(cnt_seq, len(dict_json['train']), split))

    # 训练集与验证集字典加载完成，返回生成的字典
    return dict_json


def generator(config):
    # 创建一个空 json
    dict_json = {'train': [], 'test': []}

    # 数据集路径
    rootpath = config['data']['root_path']

    # 生成训练集和初始测试集
    dict_json = generate_json(dict_json=dict_json, rootpath=os.path.join(rootpath, 'two_view_training'), split='train')
    dict_json = generate_json(dict_json=dict_json, rootpath=os.path.join(rootpath, 'two_view_testing'), split='test')

    return dict_json
