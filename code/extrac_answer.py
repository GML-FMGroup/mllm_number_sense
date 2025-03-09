import os
from utils import judge, get_dict
import argparse
import os
from configs import index, model_list, models_dir
import pandas as pd
import shutil
import json
from transformers import AutoTokenizer, AutoModel
import torch

import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info
# model_path = "Qwen/Qwen2.5-VL-72B-Instruct"
#     # default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     model_path, torch_dtype="auto", device_map="auto"
# )
# processor = AutoProcessor.from_pretrained(model_path)
model_path = models_dir + "Internvl2_5-38B"
device_map = "balanced"
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    # use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

def run_model_get(prompt):
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": prompt}
    #         ],
    #     }
    # ]
    # # Preparation for inference
    # text = processor.apply_chat_template(
    #     messages, tokenize=False, add_generation_prompt=True
    # )
    # image_inputs, video_inputs = process_vision_info(messages)
    # inputs = processor(
    #     text=[text],
    #     images=image_inputs,
    #     videos=video_inputs,
    #     padding=True,
    #     return_tensors="pt",
    # )
    # inputs = inputs.to("cuda")

    # # Inference: Generation of the output
    # generated_ids = model.generate(**inputs, max_new_tokens=500)
    # generated_ids_trimmed = [
    #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    # ]
    # output_text = processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    # return output_text[0]
        # set the max number of tiles in `max_num`
    # pixel_values = load_image(image_file, max_num=12).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=1024, do_sample=False)

    # single-image single-round conversation (单图单轮对话)
    # question = '<image>\nPlease describe the image shortly.'
    response = model.chat(tokenizer,None, prompt, generation_config)
    

    return response

    
def build_prompt(question, options, prediction):
    """
    Builds the prompt for the GPT-3.5 turbo model to match an answer with several options of a single-choice question.

    If the GPT-3.5 model is unable to find a match, it will output (z).
    Also, if the original prediction does not clearly lean towards any of the options, it will output (z).

    Parameters:
    - question: String, the question.
    - options: String, the options. E.g. ['(A)', '(B)']
    - prediction: String, the answer. E.g. '(B)'
    """
    tmpl = (
        "You are an AI assistant who will help me to match an answer with several options of a single-choice question. "
        "You are provided with a question, several options, and an answer, and you need to find which option is most similar to the answer. "
        "If the answer says things like refuse to answer, I'm sorry cannot help, etc., output (z)"
        "If the meaning of all options are significantly different from the answer, or the answer does not select any option, output (z)"\
        "Your should output one of the choices, (a),(b),(c),(d),(e) (if they are valid options), or (z)\n"
        "Example 1: \n"
        "Question: Which point is closer to the camera?\nSelect from the following choices.\nOptions: (a) Point A\n(b) Point B\n(z) Failed\nAnswer: Point B, where the child is sitting, is closer to the camera.\nYour output: (b)\n"
        "Example 2: \n"
        "Question: Which point is closer to the camera?\nSelect from the following choices.\nOptions: (a) Point A\n(b) Point B\n(z) Failed\nAnswer: I'm sorry, but I can't assist with that request.\nYour output: (z)\n"
        "Example 3: \n"
        "Question: Which point is corresponding to the reference point?\nSelect from the following choices.\nOptions: (a) Point A\n(b) Point B\n(z) Failed\nAnswer:The reference point (REF) on the first image is at the tip of the pot, which is the part used to Poke if the pots were used for that action. Looking at the second image, we need to find the part of the object that would correspond to poking.\n(a) Point A is at the tip of the spoon's handle, which is not used for poking.\n(b) Point B is at the bottom of the spoon, which is not used for poking.\n(c) Point C is on the side of the pspoonot, which is not used for poking.\n(d) Point D is at the tip of the spoon, which is not used for poking.\n\nTherefore, there is no correct answer in the choices\nYour output: (z)\n"
        "Example 4: \n"
        "Question: {}\nOptions: {}\n(z) Failed\nAnswer: {}\nYour output: "
    )
    return tmpl.format(question, options, prediction)
def get_text_option(image_id, json_data):
    text = 1
    option = 1
    for item in json_data:
        if item["id"] == image_id:
            text = item["text"]
            option = item["option"]
            break
    return text, option
def get_count(model_answer):
    options = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    found_options = set()

    # 遍历每个选项，检查是否在 model_answer 中出现
    for option in options:
        if option in model_answer:
            found_options.add(option)

    # 计算出现的选项数量
    number = len(found_options)
    return number, found_options
index = 1
file_dir_index = ["Synthetic_results","Real_results"]
file_dir_index_2 = ["Synthetic Mathematical Dataset","Real-World Dataset"]
if index == 0:
    json_file = "/data/wengtengjin/wtj_works/Only_data/Synthetic Mathematical Dataset/Synthetic.json"

else:
    json_file = "/data/wengtengjin/wtj_works/Only_data/Real-World Dataset/Real.json"

with open(json_file, 'r') as f:
    json_data = json.load(f)  # Parse the JSON data,annotations[text,image_path,option,answer]

file_dir = os.getcwd() + "/"+ file_dir_index[index]
output_dir = file_dir.replace("results", "results_final")
# if os.path.exists(output_dir):
#         shutil.rmtree(output_dir)

# os.mkdir(output_dir)

for file_name in model_list:
    file_path = os.path.join(file_dir, file_name)
    output_path = os.path.join(output_dir, file_name)
    if os.path.exists(output_path):
        continue
    if not os.path.exists(file_path):
        continue
    # Open and read the file
    results = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        # Split the line by the semicolon (';')
        parts = line.strip().split(";cut;")
        major_cls,image_id, minor_cls, model_answer, correct_answer = parts[0], parts[1],parts[2],parts[3],parts[4]
        number, found_options = get_count(model_answer)
        if number != 1:
            question, options = get_text_option(image_id, json_data['data'])
            model_answer_n = model_answer.replace(";space;","\n")
            prompt = build_prompt(question, options, model_answer_n)
            model_answer = run_model_get(prompt)
        else:
            model_answer = found_options.pop()
        result = ";cut;".join([major_cls,image_id, minor_cls, model_answer, correct_answer])

        results.append(result)
        # 将处理后的结果写入到输出文件
        with open(output_path, 'w') as output_file:
            output_file.write("\n".join(results))