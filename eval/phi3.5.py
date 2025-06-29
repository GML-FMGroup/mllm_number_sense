import os
import argparse
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from utils import run_model
from configs import Result_root, root_dir, models_dir

"""
Model example path:
    /data/wengtengjin/models/phi3_5
"""

# ------------------- Argument Parsing -------------------
parser = argparse.ArgumentParser(description="Run phi3_5 model on dataset")

parser.add_argument(
    "--model_name",
    type=str,
    default="phi3_5",
    help="Name of the model, e.g., phi3_5"
)

args = parser.parse_args()

# ------------------- Set Model Path -------------------
model_path = os.path.join(models_dir, args.model_name)
result_file_path = os.path.join(Result_root, f"{args.model_name}.json")

print(f"Loading model from: {model_path}")

# ------------------- Load Model & Processor -------------------
# Use flash attention if available; otherwise, switch to 'eager'
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation='flash_attention_2'
)

# Set number of crops; 4 for multi-frame, 16 for single-frame recommended
processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
    num_crops=4
)

print(f"Model {args.model_name} loaded successfully.")

# ------------------- Inference Function -------------------
def run_phi3_5(prompt_text, image_path):
    """
    Run phi3_5 model on a given image and prompt.
    """

    # Load image
    images = [Image.open(image_path)]

    # Create placeholder for images in prompt format
    placeholder = "<|image_1|>\n"

    messages = [
        {
            "role": "user",
            "content": placeholder + prompt_text
        }
    ]

    # Construct chat prompt
    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Prepare inputs
    inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")

    # Generation settings
    generation_args = {
        "max_new_tokens": 512,
        "temperature": 0.0,
        "do_sample": False,
    }

    # Generate output
    generated_ids = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        **generation_args
    )

    # Remove prompt tokens from output
    generated_ids_trimmed = generated_ids[:, inputs['input_ids'].shape[1]:]

    # Decode prediction
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return response

# ------------------- Execute Script -------------------
if __name__ == "__main__":
    run_model(root_dir, result_file_path, run_phi3_5)
