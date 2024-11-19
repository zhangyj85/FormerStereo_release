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


def generate_2014_json(dict_json, rootpath="/media/kunb/Dataset/SceneFlow/Driving/disparity", split="train"):

    # 生成包含 left + right + gt + K 相对路径的.json文件
    # 用于生成训练集

    # For train splits
    dict_json = dict_json
    cnt_seq = 0                                 # scence计数

    # 考虑不同曝光下的图像对, 每个场景有3组样例
    imgs_filepaths = glob.glob(os.path.join(rootpath, "**/im0.png"), recursive=True)
    imgs_filepaths.sort()
    right_image_set = ["im1.png", "im1E.png", "im1L.png"]  # 仅 2014 可用
    # 输出总样本数
    print("Total samples in Middlebury {} set: {}".format(split, len(imgs_filepaths)))
    for file_path in imgs_filepaths:
        path_scene = os.path.dirname(file_path)
        path_img0 = file_path
        path_img1 = os.path.join(path_scene, "im1.png")
        path_imgE = os.path.join(path_scene, "im1E.png")
        path_imgL = os.path.join(path_scene, "im1L.png")
        path_calib = os.path.join(path_scene, "calib.txt")
        path_disp0 = os.path.join(path_scene, "disp0.pfm")
        path_disp1 = os.path.join(path_scene, "disp1.pfm")
        dict_sample = {
            "img1": path_img0, "img2": path_img1, "img2E": path_imgE, "img2L": path_imgL,
            "calib": path_calib, "disp1": path_disp0, "disp2": path_disp1,
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

        dict_json[split].append(dict_sample)
        cnt_seq += 1

    print("Add {} samples, Total {} samples".format(cnt_seq, len(dict_json[split])))

    # 训练集与验证集字典加载完成，返回生成的字典
    return dict_json


def generate_eval_json(dict_json, rootpath="/media/kunb/Dataset/SceneFlow/Driving/disparity", split="train"):

    # 生成包含 left + right + gt + K 相对路径的.json文件
    # 用于生成训练集

    # For train splits
    dict_json = dict_json
    cnt_seq = 0                                 # scence计数

    # 考虑不同曝光下的图像对, 每个场景有3组样例
    imgs_filepaths = glob.glob(os.path.join(rootpath, "**/im0.png"), recursive=True)
    imgs_filepaths.sort()
    # 输出总样本数
    print("Total samples in Middlebury {} set: {}".format(split, len(imgs_filepaths)))
    for file_path in imgs_filepaths:
        path_scene = os.path.dirname(file_path)
        path_img0 = file_path
        path_img1 = os.path.join(path_scene, "im1.png")
        path_calib = os.path.join(path_scene, "calib.txt")
        path_disp0 = os.path.join(path_scene, "disp0GT.pfm")
        path_nocc0 = os.path.join(path_scene, "mask0nocc.png")
        dict_sample = {
            "img1": path_img0, "img2": path_img1,
            "calib": path_calib, "disp1": path_disp0, "nocc1": path_nocc0,
        }

        # 非训练集, 删除 真值键值对
        if 'train' not in split:
            del dict_sample['disp1'], dict_sample['nocc1']
            if split == "test-f":
                dict_sample['disp1'] = os.path.join(path_scene.replace('testF', 'test_ref/DLNR_32'), 'disp0pred.pfm')

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

    print("Add {} samples, Total {} samples".format(cnt_seq, len(dict_json[split])))

    # 训练集与验证集字典加载完成，返回生成的字典
    return dict_json


def generator(config):
    rootpath = config['data']['root_path']  # 数据集根路径(包含 trainingF, trainingH 等6个文件夹的父路径)
    # 创建一个空 json
    dict_json = {
        'train': [], 'val': [],
        'train-f': [], 'train-h': [], 'train-q': [],
        'test-f': [], 'test-h': [], 'test-q': []
    }
    # 生成训练集
    dict_json = generate_2014_json(dict_json=dict_json, rootpath=os.path.join(rootpath, '2014'), split='train')
    dict_json = generate_eval_json(dict_json=dict_json, rootpath=os.path.join(rootpath, 'MiddEval3/trainingF'), split='train')

    # 生成测试集
    dict_json = generate_eval_json(dict_json=dict_json, rootpath=os.path.join(rootpath, 'MiddEval3/trainingF'), split='train-f')
    dict_json = generate_eval_json(dict_json=dict_json, rootpath=os.path.join(rootpath, 'MiddEval3/trainingH'), split='train-h')
    dict_json = generate_eval_json(dict_json=dict_json, rootpath=os.path.join(rootpath, 'MiddEval3/trainingQ'), split='train-q')
    dict_json = generate_eval_json(dict_json=dict_json, rootpath=os.path.join(rootpath, 'MiddEval3/testF'), split='test-f')
    dict_json = generate_eval_json(dict_json=dict_json, rootpath=os.path.join(rootpath, 'MiddEval3/testH'), split='test-h')
    dict_json = generate_eval_json(dict_json=dict_json, rootpath=os.path.join(rootpath, 'MiddEval3/testQ'), split='test-q')
    return dict_json