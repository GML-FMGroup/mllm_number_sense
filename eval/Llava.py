from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import os
from utils import run_model
from configs import Result_root, root_dir, models_dir
import argparse
"""
/data/wengtengjin/models/llava-v1.5-7b
/data/wengtengjin/models/llava-v1.5-13b
/data/wengtengjin/models/llava-v1.6-34b
Math-llava-13b

"""
parser = argparse.ArgumentParser(description="Script for processing data")

# Add an argument for Data_number
parser.add_argument(
    "--model_name", 
    type=str, 
    default="Math-llava-13b", 
    help="llava-v1.6-34b;llava-v1.5-13b;llava-v1.5-7b;Math-llava-13b"
)

# Parse the arguments
args = parser.parse_args()

model_path = models_dir + args.model_name
print(f"{args.model_name} laoded")

file_path = os.path.join(Result_root, model_path.split("/")[-1]+".json")

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

# prompt = "What is the angle between the two line segments in the picture?"
# image_file = "https://llava-vl.github.io/static/images/view.jpg"
# image_file = r"/data/wengtengjin/Number_Sense/Chart_Game_data/Picture/Angel/image/4_19.png"

def runllava(prompt, image_file):
    args = type('Args', (), {
        "model_name":model_name,
        "tokenizer": tokenizer,
        "model": model,
        "image_processor": image_processor,
        "context_len":context_len,
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    return eval_model(args)
if __name__ == "__main__":
    run_model(root_dir, file_path, runllava)
        

