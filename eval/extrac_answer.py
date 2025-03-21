import os
import os
from configs import index, model_list, models_dir
import json
from transformers import AutoTokenizer, AutoModel
import torch

# You can use other models to extract the answer,we use InetrVL2.5_38B
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
    generation_config = dict(max_new_tokens=1024, do_sample=False)
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
    for item in json_data:
        if item["id"] == image_id:
            text = item["question"]
            option = item["option"]
            break
    return text, option

#control the Synthetic Scenario or Real-world Scenario
index = 0
save_results_dir = ["save_synthetic_output","save_real_output"]

#the json data of VisNumbench
if index == 0:
    json_file = "../datasets/Synthetic Mathematical Dataset/Synthetic.json"
else:
    json_file = "../datasets/Real-World Dataset/Real.json"

with open(json_file, 'r') as f:
    question_option_data = json.load(f)  # Parse the JSON data,annotations[text,image_path,option,answer]

file_dir = os.getcwd() + "/"+ save_results_dir[index]
output_dir = file_dir.replace("output", "prediction")

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for file_name in model_list:
    json_path = os.path.join(file_dir, file_name)
    save_json_path = os.path.join(output_dir, file_name)
    if not os.path.exists(json_path):
        continue
    # Open and read the file
    results = []
    
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    for data in json_data:
        image_id = data["image_id"]
        model_answer_n = data["predict_answer"]
        question, options = get_text_option(image_id, question_option_data)
        prompt = build_prompt(question, options, model_answer_n)
        model_prediction = run_model_get(prompt)
        
        results.append({image_id:model_prediction})
        # print(results)

    with open(save_json_path, 'w') as f:
        json.dump(results, f, indent=4)

