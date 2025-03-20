import os
from utils import judge, get_dict
import argparse
import os
from configs import index, model_list
import json

def get_correct_answer(image_id, json_data):
    for item in json_data:
        if item["id"] == image_id:
            correct_answer = item["answer"]
            break
    return correct_answer

parser = argparse.ArgumentParser(description="Script for processing data")

#control the Synthetic Scenario or Real-world Scenario
index = 0
save_results_dir = ["save_synthetic_prediction","save_real_prediction"]
#the json data of VisNumbench
if index == 0:
    json_file = "/data/wengtengjin/wtj_works/Only_data/Synthetic Mathematical Dataset/Synthetic.json"
else:
    json_file = "/data/wengtengjin/wtj_works/Only_data/Real-World Dataset/Real.json"
with open(json_file, 'r') as f:
    correct_answer_data = json.load(f)  # Parse the JSON data,
file_dir = os.getcwd() + "/"+ save_results_dir[index]
accuracies_percentage_list=[]
for file_name in model_list:
    json_path = os.path.join(file_dir, file_name)
    if not os.path.exists(json_path):
        continue
    # Open and read the file
    
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    major_dict = {}
    minor_dict = {}
    # Process each line
    for data in json_data:
        image_id = list(data.keys())[0]
        model_prediction = list(data.values())[0]
        major_cls = image_id.split("_")[-1]

        if major_cls not in major_dict:
            major_dict[major_cls] = {'total': 0, 'correct': 0}
            
        major_dict[major_cls]['total'] += 1
        correct_answer = get_correct_answer(image_id, correct_answer_data['data'])
        if judge(model_prediction, correct_answer):
            major_dict[major_cls]['correct'] += 1
    accuracies_percentage = get_dict(major_dict, file_name)
    accuracies_percentage_list.append(accuracies_percentage)

# writing excel
import csv
file_name = save_results_dir[index] + ".csv"
with open(file_name, mode='w', newline='') as file:
    if index == 1:
        writer = csv.DictWriter(file, fieldnames=['model_name', 'Angle', 'Length', 'Scale', 'Quantity', 'Depth','Volume', 'Ave'])
    else:
        writer = csv.DictWriter(file, fieldnames=['model_name', 'Angle', 'Length', 'Scale', 'Quantity', 'Depth','Area', 'Ave'])
    writer.writeheader()
    
    for accuracy in accuracies_percentage_list:
        writer.writerow(accuracy)

print(f"数据已写入 {file_name}")