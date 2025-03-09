import os
from utils import run_model
from configs import Result_root, root_dir,models_dir
import argparse

from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
"""
/data/wengtengjin/models/phi3_5
"""
parser = argparse.ArgumentParser(description="Script for processing data")

# Add an argument for Data_number
parser.add_argument(
    "--model_name", 
    type=str, 
    default="phi3_5", 
    help="phi3_5"
)

# Parse the arguments
args = parser.parse_args()
model_path = models_dir + args.model_name
print(f"{args.model_name} laoded")
file_path = os.path.join(Result_root, model_path.split("/")[-1]+".txt")



# Note: set _attn_implementation='eager' if you don't have flash_attn installed
model = AutoModelForCausalLM.from_pretrained(
  model_path, 
  device_map="cuda", 
  trust_remote_code=True, 
  torch_dtype="auto", 
  _attn_implementation='flash_attention_2'    
)

# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
processor = AutoProcessor.from_pretrained(model_path, 
  trust_remote_code=True, 
  num_crops=4
) 



def runPhi3_5(text, image_path):
    images = []
    placeholder = ""

    # Note: if OOM, you might consider reduce number of frames in this example.
    images.append(Image.open(image_path))
    placeholder += f"<|image_{1}|>\n"

    messages = [
        {"role": "user", "content": placeholder + text},
    ]

    prompt = processor.tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
    )

    inputs = processor(prompt, images, return_tensors="pt").to("cuda:0") 

    generation_args = { 
        "max_new_tokens": 1000, 
        "temperature": 0.0, 
        "do_sample": False, 
    } 

    generate_ids = model.generate(**inputs, 
    eos_token_id=processor.tokenizer.eos_token_id, 
    **generation_args
    )

    # remove input tokens 
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False)[0] 
    return response

if __name__ == "__main__":
    run_model(root_dir, file_path, runPhi3_5)