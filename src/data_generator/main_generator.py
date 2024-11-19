"""
Description: python main_generator.py for generate the json file for training and testing
Last modify: 2023.03.13
"""
import os.path
import random
import argparse
import json
import sys

from generate_sceneflow_json import generator as SceneFlowDataset
from generate_crestereo_json import generator as CREStereoDataset
from generate_kitti2015_json import generator as KITTIStereo2015
from generate_kitti2012_json import generator as KITTIStereo2012
from generate_middlebury_json import generator as Middlebury
from generate_eth3d_json import generator as ETH3D
# from generate_drivingstereo_json import generator as DrivingStereo
# from generate_oxford_json import generator as Oxford


__datasets__ = {
    "sceneflow": SceneFlowDataset,
    "kitti2015": KITTIStereo2015,
    "kitti2012": KITTIStereo2012,
    "crestereo": CREStereoDataset,
    "middlebury": Middlebury,
    "eth3d": ETH3D,
    # "drivingstereo": DrivingStereo,
    # "oxford": Oxford,
}


sys.path.append("..")   # 将上级目录添加到索引中, 使得 import 可以成功
from Options import parse_opt
parser = argparse.ArgumentParser()
parser.add_argument('--options', type=str, help='Path to the option JSON file.', default='../Options/options.json')
args = parser.parse_args()
args = parse_opt(args.options)

random.seed(args["environment"]["seed"])


def dump_json(config):

    # 数据集选择
    dset = config['data']['datasets'][0].lower()
    generate_func = __datasets__[dset]

    # 生成 json 字典集合
    dict_json = generate_func(config)

    # 对训练集打乱
    # random.seed(config['environment']['seed'])
    # random.shuffle(dict_json['train'])

    # 将最终的json文件写入硬盘
    json_root = os.path.join("..", args["data"]["data_json"])
    file_name = dset + ".json"
    file_path = os.path.join(json_root, file_name)
    if not os.path.exists(json_root):
        os.makedirs(json_root)

    f = open(file_path, 'w')
    json.dump(dict_json, f, indent=4)  # 缩进 indent = 4
    f.close()

    # 总结最终生成的文件数目
    for key in dict_json.keys():
        print("{name} subset size: {len}".format(name=key, len=len(dict_json[key])))
    print("Json file generation finished.")


if __name__ == '__main__':
    dump_json(args)