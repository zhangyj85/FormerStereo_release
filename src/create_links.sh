#!/bin/bash
# author: Yongjian Zhang
# time: 2024.07.04
# this script is used for create the soft links to the office directory

# 将数据集根路径链接到当前工作区根路径
SOURCE_DATA_ROOT="/media/zhangyj85/Dataset/Stereo_Datasets"
CURRENT_ROOT=$(dirname "$PWD")
TARGET_DATA_ROOT="${CURRENT_ROOT}/datasets"
ln -s $SOURCE_DATA_ROOT $TARGET_DATA_ROOT

# 创建并链接 checkpoint 保存根路径到工作区
SOURCE_CKPT_ROOT="/media/zhangyj85/HDD8T/Work_Spaces_Storage/ckpt/"
TARGET_CKPT_ROOT="${CURRENT_ROOT}/ckpt"
if [ ! -d $SOURCE_CKPT_ROOT ];then
    # 如果不存在数据目录下, 则新建
    mkdir -p $SOURCE_CKPT_ROOT
fi
ln -s $SOURCE_CKPT_ROOT $TARGET_CKPT_ROOT

# 创建并链接 foundation models weights 保存路径到工作区
SOURCE_REPO_ROOT="/media/zhangyj85/HDD8T/Work_Spaces_Storage/fundation_model_weights/"
TARGET_REPO_ROOT="${CURRENT_ROOT}/src/fundation_model_weights"
if [ ! -d $SOURCE_REPO_ROOT ];then
    # 如果不存在权重目录, 则新建
    mkdir -p $SOURCE_REPO_ROOT
fi
ln -s $SOURCE_REPO_ROOT $TARGET_REPO_ROOT

# rm -rf $TARGET_DATA_ROOT
# rm -rf $TARGET_CKPT_ROOT
# rm -rf $TARGET_REPO_ROOT

exit 0
