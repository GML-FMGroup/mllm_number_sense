import os
from utils import judge, get_dict
import argparse
import os
from configs import index, json_path_dir,model_list
import pandas as pd
import json
parser = argparse.ArgumentParser(description="Script for processing data")
"""
    "Llama-VL-3_2-11B.txt",
    "Llama-VL-3_2-90B.txt",
"""
model_list = [
    "llava-onevision-qwen2-72b-si-hf.txt",
    "Internvl2_5-78B.txt",
    "llava-v1.6-34b.txt",
    "GPT-4o.txt",
    "gemini-2.0-flash.txt",
    "Qwen2.5-VL-72B-Instruct.txt",]
    
    
index = 0
file_dir_index = ["Synthetic_results_final","Real_results_final"]
json_dir = ["Synthetic Mathematical Dataset","Real-World Dataset"]
file_dir = os.getcwd() + "/"+ file_dir_index[index]
json_path_dir = "/data/wengtengjin/wtj_works/Only_data/" + json_dir[index]
for json_file in os.listdir(json_path_dir):
    if json_file.endswith("json"):
        json_path = os.path.join(json_path_dir, json_file)
    else:
        continue
    with open(json_path, 'r') as f:
        json_data = json.load(f)  # Parse the JSON data,annotations[text,image_path,option,answer]

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
        minor_cls,image_id, _, model_answer, correct_answer = parts[0], parts[1],parts[2],parts[3],parts[4]
        major_cls = None
        for item in json_data["data"]:
            if item["id"] == image_id:
                major_cls = item["task_class"]
                break

        # major_cls = major_cls.split("/")[0]
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
    # print(accuracies_percentage)
    
import csv
# CSV 文件名
file_name = file_dir_index[index] + "different.csv"

# 打开文件并写入数据
with open(file_name, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['model_name', 'range estimation', 'value estimation', 'value comparison', 'multiplicative estimation',"Ave"])
    # 写入表头
    writer.writeheader()
    
    # 写入每个字典的数据
    for accuracy in accuracies_percentage_list:
        writer.writerow(accuracy)

print(f"数据已写入 {file_name}")