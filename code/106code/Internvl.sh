#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 初始化conda
source ~/miniconda3/etc/profile.d/conda.sh  # 替换成你的 Conda 安装路径
conda activate intervl

python Internvl2.py --model_name=Internvl-2B >> internvl.txt 2>&1 &
# 等待第一个进程完成
wait $!
python Internvl2.py --model_name=Internvl-8B >> internvl.txt 2>&1 &
wait $!
python Internvl2.py --model_name=Internvl2_5-8B >> internvl.txt 2>&1 &
wait $!
python Internvl2.py --model_name=Internvl2_5-38B >> internvl.txt 2>&1 &
wait $!
python Internvl2.py --model_name=Internvl2_5-78B >> internvl.txt 2>&1 &
wait $!
python Internvl2.py --model_name=InternVL2-8B-MPO >> internvl.txt 2>&1 &
wait $!

# 可选：退出conda环境
conda deactivate