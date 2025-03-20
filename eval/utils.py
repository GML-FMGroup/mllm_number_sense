from pathlib import Path
import json
import os
from configs import json_path_dir


def build_prompt(question, option):
    prompt = f"""
        Question: {question}
        Options: {option}
        Answer the question based on the most likely options."""
    return prompt

def get_json_path():
    json_files = [str(file) for file in Path(json_path_dir).rglob('*.json')]
    return json_files
        
def write_json(save_json_path, save_json_data):
    if os.path.exists(save_json_path):
        # Read existing data
        with open(save_json_path, 'r') as f:
            data = json.load(f)
        # Assume the file content is a list
        if isinstance(data, list):
            # Append new data
            data.append(save_json_data)
        else:
            # If not a list, create a new list containing existing data and new data
            data = [data, save_json_data]
    else:
        # If the file does not exist, create a list containing new data
        data = [save_json_data]

    # Write updated data back to the file
    with open(save_json_path, 'w') as f:
        json.dump(data, f, indent=4)

def run_model(root_dir, save_json_path, model_run):
    '''
    '''
    json_paths = get_json_path()
    for json_file in json_paths:
        with open(json_file, 'r') as f:
            json_data = json.load(f)  # Parse the JSON data,annotations[text,image_path,option,answer]

        for data in json_data["data"]:
            
            image_id = data['id']
            question = data['text']
            image_path = data["image_path"]
            # image_path = data["image_path"].replace("Number_Sense","Only_data")
            image_file = os.path.join(root_dir, image_path)
            option = data["option"]
            answer = data['answer']
            prompt = build_prompt(question, option)
            predict_answer = model_run(prompt, image_file)
            save_json_data={
                "image_id":image_id,
                "predict_answer":predict_answer,
                "answer":answer
            }
            # print(predict_answer)
            write_json(save_json_path, save_json_data)

def judge(model_prediction, correct_answer):
    if model_prediction == correct_answer[:3]:
        return True
    else:
        return False
    
def get_dict(major_dict, file_name):
    # 计算每个类别的准确率
    accuracies_percentage = {'model_name':file_name[:-4]}
    total_correct = 0  # 用于计算总的正确数
    total_total = 0    # 用于计算总的样本数

    for category, values in major_dict.items():
        total = values['total']
        correct = values['correct']
        accuracy = (correct / total) * 100 if total != 0 else 0  # 计算百分比
        accuracies_percentage[category] = f"{accuracy:.2f}"  # 格式化为百分比并保留两位小数
        
        # 累加总的正确数和样本数
        total_correct += correct
        total_total += total

    # 计算平均准确率
    average_accuracy = (total_correct / total_total) * 100 if total_total != 0 else 0
    accuracies_percentage['Ave'] = f"{average_accuracy:.2f}"  # 将平均值添加到字典中，格式化为百分比
    return accuracies_percentage