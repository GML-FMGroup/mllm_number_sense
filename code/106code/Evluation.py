import os
from utils import judge, get_results
import argparse
import os
parser = argparse.ArgumentParser(description="Script for processing data")
"""
Intervl-2B.txt
llava-v1.5-7b.txt
Qwen2-VL-2B-Instruct.txt
Qwen2-VL-7B-Instruct.txt
Qwen2-VL-72B-Instruct.txt
Intervl2_5-8B.txt
Internvl2_5-38B.txt
Llama-VL-3_2-11B.txt
"""
# Add an argument for Data_number
parser.add_argument(
    "--file_name", 
    type=str, 
    default="Qwen2-VL-2B-Instruct.txt", 
    help="file_name"
)
index = 0
file_dir = ["Synthetic_results","Real_results"]
# Parse the arguments
args = parser.parse_args()

file_dir = os.getcwd() + "/"+ file_dir[index]

# Path to the txt file
file_name = args.file_name
print(file_name)


file_path = os.path.join(file_dir, file_name)
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
    major_cls, minor_cls, model_answer, correct_answer = parts[0], parts[1],parts[2],parts[3]
    major_cls = major_cls.split("/")[0]
    if major_cls not in major_dict:
        major_dict[major_cls] = {'total': 0, 'correct': 0}
    if minor_cls not in minor_dict:
        minor_dict[minor_cls] = {'total': 0, 'correct': 0}
    
    major_dict[major_cls]['total'] += 1
    minor_dict[minor_cls]['total'] += 1
    if judge(model_answer, correct_answer):
        major_dict[major_cls]['correct'] += 1
        minor_dict[minor_cls]['correct'] += 1


get_results(major_dict)
print("__________________________")
get_results(minor_dict)


