import base64
from PIL import Image
from pathlib import Path
import json
import os
from configs import json_path_dir
import random

space_str = ";space;"
# other_text = ".You only need to estimate from the visual information and do not need to do any mathematical reasoning."
# other_text = "Select from the following choices."
other_text = ".Please output answer from options directly."
# other_text = ".Please output (a),(b),(c),(d),(e) from options directly."

def build_prompt(text, option):
    # prompt = f"""
    #     You are a Visual Question Answering (VQA) model.
    #     Question: {text}
    #     Options: {option}
    #     Answer the question based on the most likely options."""
    #     # Provide only the letter corresponding to your choice as the answer (e.g., '(a)', '(b)', '(c)', '(d)', '(e)')."""
    prompt = f"""
    You are a Visual Question Answering (VQA) model.
    Question: {text}
    Options: {option}
    Please think and answer the question based on the most likely options."""
    return prompt


def get_json_path():
    json_files = [str(file) for file in Path(json_path_dir).rglob('*.json')]
    return json_files

# Base64 编码格式
def encode_image(image_path, size = 256):
    # 打开图像
    with Image.open(image_path) as img:
        # 缩放图像，使其最长边为 256，保持纵横比
        img.thumbnail((size, size))  # 只会缩放至最大边为 256 像素
        # 将图像转换为字节数据并进行 Base64 编码
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
        
def write_txt(file_path, text_output):
    # 写入文件
    with open(file_path, 'a') as file:
        file.write(f"{text_output}\n")
        
        
def read_txt(file_path):
        #read the txt file,get the last position
    image_id_list = set()
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            # Split the line by the semicolon (';')
            parts = line.strip().split(";cut;")
            major_cls,image_id, minor_cls, model_answer, correct_answer = parts[0], parts[1],parts[2],parts[3],parts[4]
            image_id_list.add((image_id))
    return list(image_id_list)

def Is_full(file_path):
    image_id_list = read_txt(file_path)
    json_paths = get_json_path()

    for json_file in json_paths:
        with open(json_file, 'r') as f:
            json_data = json.load(f)
    if len(image_id_list) == len(json_data["data"]):
        return True
    else:
        return False


def run_model(root_dir, file_path, model_run):

    image_id_list = read_txt(file_path)
    print(image_id_list)
    #先得到json数据
    json_paths = get_json_path()
    # print(json_paths)
    # json_paths = random.shuffle(json_paths)
    for json_file in json_paths:
        with open(json_file, 'r') as f:
            json_data = json.load(f)  # Parse the JSON data,annotations[text,image_path,option,answer]

        for data in json_data["data"]:
            image_id = data['id']
            if image_id in image_id_list:
                continue
            text = data['text']
        
            image_path = data["image_path"]
            # image_path = data["image_path"].replace("Number_Sense","Only_data")
            image_file = os.path.join(root_dir, image_path)
            Major_cls = data['Major_categories']
            # if Major_cls not in ["Depth","Quantity", "Area","Length"]:
                # continue
            Minor_cls = data['Minor_category']
            # if Minor_cls not in ["bottle_water_compare", "box-bottle","bike_count"]:
            #     continue
            option = data["option"]
            answer = data['answer']
            # prompt = text + "" + option + other_text
            prompt = build_prompt(text, option)
            # print(prompt)
            # prompt = text + option
            # prompt = text + other_text + option
            predict_answer = model_run(prompt, image_file)
            output_text = f"{Major_cls};cut;{image_id};cut;{Minor_cls};cut;{predict_answer};cut;{answer}"
            output_text = output_text.replace("\n",space_str)
            write_txt(file_path, output_text)


def judge(model_answer, correct_answer,file_name):
    if model_answer == correct_answer[:3]:
        return True
    else:
        return False
    #     # 使用正则表达式提取括号中的内容
    # # ture_answer = re.match(r"\((.*)\)", correct_answer)
    # # print(model_answer.count("(a)"),model_answer.count("(b)"),model_answer.count("(c)"),model_answer.count("(d)"))
    # model_answer_raw = model_answer
    # true_answer = correct_answer[:3]
    # true_answer_text = correct_answer[4:]
    # # 使用 set 来存储出现的选项
    # # 定义需要统计的选项
    # options = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    # found_options = set()

    # # 遍历每个选项，检查是否在 model_answer 中出现
    # for option in options:
    #     if option in model_answer:
    #         found_options.add(option)

    # # 计算出现的选项数量
    # number = len(found_options)
    
    # # if number > 1:
    # #     # print(model_answer,file_name)
    # #     model_answer_short = model_answer.split(space_str)[-1]
    # #     found_options_short = set()

    # #     # 遍历每个选项，检查是否在 model_answer 中出现
    # #     for option in options:
    # #         if option in model_answer_short:
    # #             found_options_short.add(option)
    # #     number = len(found_options_short)
    # #     print(number, file_name)

    # #     # 计算出现的选项数量
    # # if number == 1 and true_answer in model_answer:
    # #         return True

    # # if space_str not in model_answer_raw:
    # #     model_answer_short = model_answer.replace("(", "").replace(")", "").lower()[0]
    # #     if model_answer_short in true_answer:
    # #         return True

    # return False
    
    # #     # 先将 model_answer 使用空格分隔
    # # model_answers = model_answer.split()
    


def get_results(results_dict):
    total_correct = 0
    total_samples = 0
    # 计算并输出每个 cls 的准确率
    for cls, stats in results_dict.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"total:{stats['total']};correct_num:{stats['correct']};Accuracy for {cls}: {accuracy * 100:.2f}%")
        
        # 累加总的正确数和总样本数
        total_correct += stats['correct']
        total_samples += stats['total']

    # 计算所有类别的平均准确率
    average_accuracy = total_correct / total_samples if total_samples > 0 else 0

    # 输出平均准确率
    print(f"Average Accuracy: {average_accuracy * 100:.2f}%")
    
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


