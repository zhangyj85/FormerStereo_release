#!/bin/bash

# this script uses a single node (`NUM_NODES=1`) with 4 GPUs (`NUM_GPUS_PER_NODE=4`)
export NUM_NODES=1
export NODE_RANK=0
export NUM_GPUS_PER_NODE=1
export CUDA_VISIBLE_DEVICES=1
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=14900

# nproc_per_node: 单个节点上的进程数, 与 GPU 数量一致 (需要同时修改 CUDA_VISIBLE_DEVICES 变量)
# nnodes:         节点总数, 单机多卡时节点数为 1
# node_rank:      当前节点的序号, 节点总数为 1, 故序号为 0
# use_env:		  为当前进程自动分配 LOCAL_RANK 序号, 并使用默认的通讯端口和地址, MASTER ADDR:PORT = 127.0.0.1:29500

python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --use_env \
    ddp_main.py \
    --options './Options/options.json'
