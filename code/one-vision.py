import os
from utils import run_model, Is_full
from configs import Result_root, root_dir, models_dir
import argparse
from transformers import pipeline, AutoProcessor
from PIL import Image    

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
"""
/data/wengtengjin/models/llava-onevision-qwen2-72b-si-hf
"""

parser = argparse.ArgumentParser(description="Script for processing data")

# Add an argument for Data_number
parser.add_argument(
    "--model_name", 
    type=str, 
    default="llava-onevision-qwen2-7b-ov-hf", 
    help="llava-onevision-qwen2-72b-si-hf, llava-onevision-qwen2-7b-ov-hf"
)

# Parse the arguments
args = parser.parse_args()
if args.model_name == "llava-onevision-qwen2-7b-ov-hf":
    model_path = models_dir + args.model_name
else:
    model_path = "llava-hf/llava-onevision-qwen2-72b-si-hf"
file_path = os.path.join(Result_root, model_path.split("/")[-1]+".txt")
if not Is_full(file_path):
    full_file = 0
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        device_map = "auto"
    )

    processor = AutoProcessor.from_pretrained(model_path)
    print(f"{args.model_name} laoded")

else:
    full_file = 1
    print(f"{args.model_name} already writing")




def runllava_one(text, image_path):
    image = Image.open(image_path)
    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)

    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    return processor.decode(output[0][2:], skip_special_tokens=True).split("assistant")[-1]


if __name__ == "__main__":
    if full_file == 0:
        run_model(root_dir, file_path, runllava_one)

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
