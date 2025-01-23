#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 初始化conda
source ~/miniconda3/etc/profile.d/conda.sh  # 替换成你的 Conda 安装路径
conda activate llava

python Llava.py --model_name=llava-v1.5-7b >> llava.txt 2>&1 &
# 等待第一个进程完成
wait $!
python Llava.py --model_name=llava-v1.5-13b >> llava.txt 2>&1 &
# 等待第一个进程完成
wait $!
python Llava.py --model_name=llava-v1.6-34b >> llava.txt 2>&1 &
# 等待第一个进程完成
wait $!



# 可选：退出conda环境
conda deactivate
