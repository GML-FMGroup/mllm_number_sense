import base64
from PIL import Image
from pathlib import Path
import json
import os
from configs import json_path_dir


space_str = ";space;"
# other_text = ".You only need to estimate from the visual information and do not need to do any mathematical reasoning."
# other_text = "Select from the following choices."
other_text = ".Please output answer directly."


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
        
def run_model(root_dir, file_path, model_run):
    
    #先得到json数据
    json_paths = get_json_path()
    # print(json_paths)
    for json_file in json_paths:
        with open(json_file, 'r') as f:
            json_data = json.load(f)  # Parse the JSON data,annotations[text,image_path,option,answer]

        for data in json_data["data"]:
            text = data['text']
        
            image_path = data["image_path"]
            # image_path = data["image_path"].replace("Number_Sense","Only_data")
            image_file = os.path.join(root_dir, image_path)
            Major_cls = data['Major_categories']
            Minor_cls = data['Minor_category']
            option = data["option"]
            answer = data['answer']
            prompt = text + "" + option + other_text
            # prompt = text + option
            # prompt = text + other_text + option
            predict_answer = model_run(prompt, image_file)
            output_text = f"{Major_cls};cut;{Minor_cls};cut;{predict_answer};cut;{answer}"
            output_text = output_text.replace("\n",space_str)
            write_txt(file_path, output_text)


def judge(model_answer, correct_answer):
        # 使用正则表达式提取括号中的内容
    # ture_answer = re.match(r"\((.*)\)", correct_answer)
    # print(model_answer.count("(a)"),model_answer.count("(b)"),model_answer.count("(c)"),model_answer.count("(d)"))
    model_answer_raw = model_answer
    if space_str in model_answer:
        model_answer = model_answer.split(space_str)[-1]
        # print(model_answer)
    true_answer = correct_answer.split(" ")[0]
    true_answer_text = correct_answer.split(" ")[1]
    number = model_answer.count("(a)") + model_answer.count("(b)") + model_answer.count("(c)") + model_answer.count("(d)")
    if number == 1 and true_answer in model_answer or true_answer_text in model_answer:
            return True
        #第一个字符
    if space_str not in model_answer_raw:
        model_answer_short = model_answer.replace("(", "").replace(")", "").lower()[0]
        if model_answer_short in true_answer:
            return True

    return False
    
    #     # 先将 model_answer 使用空格分隔
    # model_answers = model_answer.split()
    


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
    
