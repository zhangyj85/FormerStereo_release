"""
    工程项目: XYZ 主动双目深度探测, 生成数据 sample 所需路径
    张勇健， 17308223， zhangyj85@mail2.sysu.edu.cn
    last modify: 2023.03.13
    ======================================================================

    This script generates a json file for SceneFlow dataset.
    python utils/generate_sceneflow_json.py
"""

import os
import glob
import random
random.seed(0)


def generate_json(dict_json, rootpath="/media/kunb/Dataset/KITTI2012", split="train"):
    # 生成包含 left + right + gt + K 相对路径的.json文件
    # 用于生成训练集

    # For train splits
    dict_json = dict_json
    cnt_seq = 0  # scence计数

    imgs_filepaths = glob.glob(os.path.join(rootpath, split + "ing", "colored_0/*.png"), recursive=True)
    imgs_filepaths.sort()
    print("Total samples in KITTI2012: {}".format(len(imgs_filepaths)))
    for file_path in imgs_filepaths:
        filename = os.path.basename(file_path)
        index = filename[:6]
        path_img1 = file_path
        path_img2 = file_path.replace('colored_0', 'colored_1')
        path_disp = file_path.replace('colored_0', 'disp_occ')
        path_mask = file_path.replace('colored_0', 'disp_noc')
        path_calib = os.path.join(rootpath, split + "ing", "calib", index + ".txt")
        dict_sample = {
            "img1": path_img1,
            "img2": path_img2,
            "disp": path_disp,
            "mask": path_mask,
            "calib": path_calib
        }

        if split=="test":          # 非训练集, 不存在真值, 删除真值路径
            del dict_sample['disp'], dict_sample["mask"]
            dict_sample['disp'] = path_disp.replace('testing', 'testing_ref/IGEV_32')

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

    print("Add {} samples, Total {} samples in {} set".format(cnt_seq, len(dict_json[split]), split))

    # 训练集与验证集字典加载完成，返回生成的字典
    return dict_json



def generator(config):

    # 创建一个空 json
    dict_json = {'train': [], 'val': [], 'test': []}

    # 数据集路径
    rootpath = config['data']['root_path']

    # 生成训练集和初始测试集
    dict_json = generate_json(dict_json=dict_json, rootpath=rootpath, split="train")
    dict_json = generate_json(dict_json=dict_json, rootpath=rootpath, split="test")

    return dict_json
