#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 初始化conda
source ~/miniconda3/etc/profile.d/conda.sh  # 替换成你的 Conda 安装路径
conda activate llama

python Llama.py --model_name=Llama-VL-3_2-11B >> llama.txt 2>&1 &
# 等待第一个进程完成
wait $!
python Llama.py --model_name=Llama-VL-3_2-90B >> llama.txt 2>&1 &
# 等待第一个进程完成
wait $!


# 可选：退出conda环境
conda deactivate
