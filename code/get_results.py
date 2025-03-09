import os
from utils import judge, get_dict
import argparse
import os
from configs import index, model_list
import pandas as pd
parser = argparse.ArgumentParser(description="Script for processing data")
"""
    "Llama-VL-3_2-11B.txt",
    "Llama-VL-3_2-90B.txt",
"""
# model_list = [
#     "phi3_5.txt",
#     "llava-v1.5-7b.txt",
#     "llava-v1.5-13b.txt",
#     "llava-v1.6-34b.txt",
#     "Qwen2-VL-2B-Instruct.txt",
#     "Qwen2-VL-7B-Instruct.txt",
#     "Qwen2-VL-72B-Instruct.txt",
#     "llava-onevision-qwen2-72b-si-hf.txt",
#     "Internvl-2B.txt",
#     "Internvl-8B.txt",
#     "InternVL2-8B-MPO.txt",
#     "Internvl2_5-8B.txt",
#     "Internvl2_5-38B.txt",
#     "Internvl2_5-78B.txt",
#     "gemini-1.5-flash.txt",
#     "gemini-1.5-pro.txt"
# ]
index = 0
file_dir_index = ["Synthetic_results_final","Real_results_final"]
file_dir = os.getcwd() + "/"+ file_dir_index[index]
accuracies_percentage_list=[]
for file_name in model_list:
    file_path = os.path.join(file_dir, file_name)
    if not os.path.exists(file_path):
        continue
    # Open and read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # 用于存储每个 cls 的统计数据
    major_dict = {}
    minor_dict = {}
    # Process each line
    for line in lines:
        # Split the line by the semicolon (';')
        parts = line.strip().split(";cut;")
        major_cls,image_id, minor_cls, model_answer, correct_answer = parts[0], parts[1],parts[2],parts[3],parts[4]
        major_cls = major_cls.split("/")[0]
        if major_cls not in major_dict:
            major_dict[major_cls] = {'total': 0, 'correct': 0}
        if minor_cls not in minor_dict:
            minor_dict[minor_cls] = {'total': 0, 'correct': 0}
        
        major_dict[major_cls]['total'] += 1
        minor_dict[minor_cls]['total'] += 1
        if judge(model_answer, correct_answer,file_name):
            major_dict[major_cls]['correct'] += 1
            minor_dict[minor_cls]['correct'] += 1
    accuracies_percentage = get_dict(major_dict, file_name)
    accuracies_percentage_list.append(accuracies_percentage)
    
import csv
# CSV 文件名
file_name = file_dir_index[index] + ".csv"

# 打开文件并写入数据
with open(file_name, mode='w', newline='') as file:
    if index == 1:
        writer = csv.DictWriter(file, fieldnames=['model_name', 'Angle', 'Length', 'Scale', 'Quantity', 'Depth','Volume', 'Ave'])
    else:
        writer = csv.DictWriter(file, fieldnames=['model_name', 'Angle', 'Length', 'Scale', 'Quantity', 'Depth','Area', 'Ave'])
    # 写入表头
    writer.writeheader()
    
    # 写入每个字典的数据
    for accuracy in accuracies_percentage_list:
        writer.writerow(accuracy)

print(f"数据已写入 {file_name}")