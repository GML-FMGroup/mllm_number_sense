import os
import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

from utils import run_model
from configs import Result_root, root_dir, models_dir

"""
Supported model paths:
- /data/wengtengjin/models/llava-v1.5-7b
- /data/wengtengjin/models/llava-v1.5-13b
- /data/wengtengjin/models/llava-v1.6-34b
- /data/wengtengjin/models/Math-llava-13b
"""

# ------------------- Argument Parsing -------------------
parser = argparse.ArgumentParser(description="Run LLaVA model on dataset")

parser.add_argument(
    "--model_name",
    type=str,
    default="llava-v1.5-7b",
    help="Choose model: llava-v1.5-7b | llava-v1.5-13b | llava-v1.6-34b | Math-llava-13b"
)

args = parser.parse_args()

# ------------------- Resolve Paths -------------------
model_path = os.path.join(models_dir, args.model_name)
result_file_path = os.path.join(Result_root, f"{args.model_name}.json")

print(f"Loading model: {args.model_name} from {model_path}")

# ------------------- Load LLaVA Model -------------------
model_name = get_model_name_from_path(model_path)

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name
)

print(f"Model {args.model_name} loaded successfully.")

# ------------------- Inference Function -------------------
def run_llava(prompt_text, image_file):
    """
    Run LLaVA model on given prompt and image.
    Uses eval_model with dynamically constructed Args object.
    """

    # Dynamically mimic argparse.Namespace object
    args_obj = type('Args', (), {
        "model_name": model_name,
        "tokenizer": tokenizer,
        "model": model,
        "image_processor": image_processor,
        "context_len": context_len,
        "query": prompt_text,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    return eval_model(args_obj)

# ------------------- Run Inference -------------------
if __name__ == "__main__":
    run_model(root_dir, result_file_path, run_llava)
