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
import copy



# read all lines in a file
def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]  # rstrip() 删除 string 字符串末尾的指定字符，默认为空白符，包括空格、换行符、回车符、制表符
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


def generate_json(dict_json, rootpath="/media/kunb/Dataset/CREStereo/stereo_trainset/crestereo"):
    # 生成包含 left + right + gt 相对路径的.json文件
    # 用于生成训练集

    # For train splits
    dict_json = dict_json
    cnt_seq = 0  # scence计数

    imgs_filepaths = glob.glob(os.path.join(rootpath, "**/*_left.jpg"), recursive=True)
    print("Total samples in CREStereo: {}".format(len(imgs_filepaths)))
    for idx in range(len(imgs_filepaths)):
        path_img1 = imgs_filepaths[idx]
        prefix = path_img1[: path_img1.rfind("_")]
        path_img2 = prefix + "_right.jpg"
        path_disp = prefix + "_left.disp.png"
        path_disp2 = prefix + "_right.disp.png"
        dict_sample = {
            "img1": path_img1,
            "img2": path_img2,
            "disp1": path_disp,
            "disp2": path_disp2,
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


def generate_json_subset(dict_json, split, sample_num):
    # 从输入的 dict_json 中划分一部分数据到 split 分支
    # 注意采用深拷贝, 否则移除的过程改变了次序
    list_pairs = copy.deepcopy(dict_json['train'])
    random.shuffle(list_pairs)
    assert sample_num <= len(list_pairs), "There is not enough sample in the origin json dictionary!"
    list_pairs = list_pairs[:sample_num]
    dict_json[split] = dict_json[split] + list_pairs

    # 在获取完子集后，需要对原数据集部分进行改写，将子集样例从原数据集中删除
    for i in range(len(list_pairs)):
        dict_json['train'].remove(list_pairs[i])

    print(split, "split : Divide {} samples".format(len(list_pairs)))

    return dict_json


def generator(config):
    # 创建一个空 json
    dict_json = {'train': [], 'val': [], 'test': []}

    # 数据集路径
    rootpath = config['data']['root_path']

    # 生成训练集和初始测试集
    dict_json = generate_json(dict_json=dict_json, rootpath=rootpath)
    # dict_json = generate_json_subset(dict_json, 'val', sample_num=4000)

    return dict_json
