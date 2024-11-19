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
from tqdm import tqdm
import glob


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


def generate_json(dict_json, rootpath="/media/kunb/Dataset/SceneFlow/Driving/disparity"):

    # 生成包含 left + right + gt + K 相对路径的.json文件
    # 用于生成训练集

    # For train splits
    dict_json = dict_json
    cnt_seq = 0                                 # scence计数

    disp_filepaths = walk_filenames(rootpath, endpoint="pfm")
    disp_filepaths.sort()
    print("Total samples in SceneFlow: {}".format(len(disp_filepaths)))
    for file_path in tqdm(disp_filepaths):
        frame_idx = file_path.split("/")[-1].rstrip(".pfm")
        [head_path, tail_path] = file_path.split("/disparity/")
        tail_path = tail_path[:-14] # 去掉 "/left/xxxx.pfm"
        path_disp = file_path
        path_disp2 = head_path + "/disparity/" + tail_path + "/right/" + frame_idx + ".pfm"
        path_img1 = head_path + "/frames_finalpass/" + tail_path + "/left/"  + frame_idx + ".png"
        path_img2 = head_path + "/frames_finalpass/" + tail_path + "/right/" + frame_idx + ".png"
        path_calib = head_path + "/camera_data/" + tail_path + "/camera_data.txt"
        dict_sample = {
            "frame_idx": frame_idx,
            "img1": path_img1,
            "img2": path_img2,
            "disp1": path_disp,
            "disp2": path_disp2,
            "calib": path_calib
        }

        # 验证路径字典中的每个元素（value）是否存在
        flag_valid = True
        for key in dict_sample:
            if key == "frame_idx":
                continue
            flag_valid &= os.path.exists(dict_sample[key])
            if not flag_valid:
                print("Invalid key: {}, path: {}".format(key, dict_sample[key]))
                break

        # 检查外参是否记录在 calib 文件中.
        lines = read_all_lines(dict_sample["calib"])
        if "Frame {}".format(int(dict_sample["frame_idx"])) not in lines:
            # print("Frame {} without calibration. Path: {}".format(int(dict_sample["frame_idx"]), dict_sample["calib"]))
            dict_sample["calib"] = None

        if not flag_valid:
            continue

        dict_json['train'].append(dict_sample)
        cnt_seq += 1

    print("Add {} samples, Total {} samples".format(cnt_seq, len(dict_json['train'])))

    # 训练集与验证集字典加载完成，返回生成的字典
    return dict_json


def generate_json_subset(dict_json, split):

    # 从输入的 dict_json 中划分一部分数据到 split 分支, 由于 sceneflow 上在 Flything3D 上划分了测试集, 这里直接沿用
    # 注意采用深拷贝, 否则移除的过程改变了次序
    list_pairs = copy.deepcopy(dict_json['train'])
    for sample in list_pairs:
        if 'TEST' in sample["disp1"]:
            dict_json[split].append(sample)
            dict_json['train'].remove(sample)

    print(split, "split : Divide {} samples".format(len(list_pairs)))

    return dict_json


def generator(config):

    rootpath = config['data']['root_path']              # 数据集根路径
    dict_json = {'train': [], 'val': [], 'test': []}    # 创建一个空 json

    # 生成训练集和初始测试集
    dict_json = generate_json(dict_json=dict_json, rootpath=rootpath)
    dict_json = generate_json_subset(dict_json, 'val')

    return dict_json