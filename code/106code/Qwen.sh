#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 初始化conda
source ~/miniconda3/etc/profile.d/conda.sh  # 替换成你的 Conda 安装路径
conda activate GSAM

# 启动第一个 Python 进程并在后台运行，将输出追加到 Qwen2B.txt
python Qwen.py --model_name=Qwen2-VL-2B-Instruct >> Qwen.txt 2>&1 &
# 等待第一个进程完成
wait $!

# 启动第二个 Python 进程并在后台运行，将输出追加到 Qwen7B.txt
python Qwen.py --model_name=Qwen2-VL-7B-Instruct >> Qwen.txt 2>&1 &
# 等待第二个进程完成
wait $!

# 启动第三个 Python 进程并在后台运行，将输出追加到 Qwen72B.txt
python Qwen.py --model_name=Qwen2-VL-72B-Instruct >> Qwen.txt 2>&1 &
# 等待第三个进程完成
wait $!

# 可选：退出conda环境
conda deactivate
