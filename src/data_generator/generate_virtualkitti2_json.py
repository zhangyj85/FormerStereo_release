"""
    工程项目: XYZ 主动双目深度探测, 生成数据 sample 所需路径
    张勇健， 17308223， zhangyj85@mail2.sysu.edu.cn
    last modify: 2023.03.13
    ======================================================================

    This script generates a json file for SceneFlow dataset.
    python utils/generate_sceneflow_json.py
"""

import os
import copy


# read all lines in a file
def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]   # rstrip() 删除 string 字符串末尾的指定字符，默认为空白符，包括空格、换行符、回车符、制表符
    return lines


def walk_filenames(path, endpoint=None):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            # 以左视图为基准获取 samples
            if file_path.endswith(endpoint) and "right" not in root.split('/')[-1]:
                file_list.append(file_path)

    return file_list


def generate_json(dict_json, rootpath="/media/zhangyj85/Dataset/Stereo Datasets/VirtualKITTI2/"):

    # 生成包含 left + right + gt + K 相对路径的.json文件
    # 用于生成训练集

    # For train splits
    dict_json = dict_json
    cnt_seq = 0                                 # scence计数

    dep_filepaths = walk_filenames(os.path.join(rootpath, "depth"), endpoint="png")
    print("Total samples in VirtualKITTI2: {}".format(len(dep_filepaths)//2))
    for file_path in dep_filepaths:
        if "Camera_1" in file_path:
            continue    # 避免重复生成数据
        path_dep1 = file_path
        path_dep2 = file_path.replace("Camera_0", "Camera_1")
        path_img1 = path_dep1.replace("depth", "rgb").replace("png", "jpg")
        path_img2 = path_dep2.replace("depth", "rgb").replace("png", "jpg")
        dict_sample = {
            "img1": path_img1,
            "img2": path_img2,
            "dep1": path_dep1,
            "dep2": path_dep2,
        }

        # 验证路径字典中的每个元素（value）是否存在
        flag_valid = True
        for key in dict_sample:
            flag_valid &= os.path.exists(dict_sample[key])
            if not flag_valid:
                print("Invalid key: {}, path: {}".format(key, dict_sample[key]))
                break

        if not flag_valid:
            continue

        dict_json['train'].append(dict_sample)
        cnt_seq += 1

    print("Add {} samples, Total {} samples".format(cnt_seq, len(dict_json['train'])))

    # 训练集与验证集字典加载完成，返回生成的字典
    return dict_json


# def generate_json_subset(dict_json, split):
#
#     # 从输入的 dict_json 中划分一部分数据到 split 分支, 由于 sceneflow 上在 Flything3D 上划分了测试集, 这里直接沿用
#     # 注意采用深拷贝, 否则移除的过程改变了次序
#     list_pairs = copy.deepcopy(dict_json['train'])
#     for sample in list_pairs:
#         if 'TEST' in sample["disp1"]:
#             dict_json[split].append(sample)
#             dict_json['train'].remove(sample)
#
#     print(split, "split : Divide {} samples".format(len(list_pairs)))
#
#     return dict_json


def generator(config):

    rootpath = config['data']['root_path']              # 数据集根路径
    dict_json = {'train': [], 'val': []}                # 创建一个空 json

    # 生成训练集和初始测试集
    dict_json = generate_json(dict_json=dict_json, rootpath=rootpath)
    # dict_json = generate_json_subset(dict_json, 'val')

    return dict_json
