import os
from transformers import MllamaForConditionalGeneration, AutoProcessor
from utils import run_model
from configs import Result_root, root_dir,models_dir
import argparse
import torch
from PIL import Image
"""
/data/wengtengjin/models/Llama-VL-3_2-11B
/data/wengtengjin/models/Llama-VL-3_2-90B
"""

parser = argparse.ArgumentParser(description="Script for processing data")

# Add an argument for Data_number
parser.add_argument(
    "--model_name", 
    type=str, 
    default="Llama-VL-3_2-11B", 
    help="Llama-VL-3_2-11B;Llama-VL-3_2-90B;Llama-3.2V-11B-cot"
)
# Parse the arguments
args = parser.parse_args()
if "cot" in args.model_name:
    model_path = "Xkev/Llama-3.2V-11B-cot"
else:
    model_path = models_dir + args.model_name
print(f"{args.model_name} laoded")

file_path = os.path.join(Result_root, model_path.split("/")[-1]+".json")

model = MllamaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_path)

def runLlama(text, image_path):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": text},
            ],
        }
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image = Image.open(image_path)
    inputs = processor(image, input_text, return_tensors="pt").to(model.device)

    output = model.generate(**inputs, max_new_tokens=1000)
    text =  processor.decode(output[0])
    return text.split("|end_header_id|")[-1]

if __name__ == "__main__":
    run_model(root_dir, file_path, runLlama)