import os
import argparse
from PIL import Image
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor

from utils import run_model
from configs import Result_root, root_dir, models_dir

"""
Model path examples:
- /data/wengtengjin/models/Llama-VL-3_2-11B
- /data/wengtengjin/models/Llama-VL-3_2-90B
- Xkev/Llama-3.2V-11B-cot
"""

# ------------------- Argument Parsing -------------------
parser = argparse.ArgumentParser(description="Run LLaMA-VL model on dataset")

parser.add_argument(
    "--model_name", 
    type=str, 
    default="Llama-VL-3_2-11B", 
    help="Model name: Llama-VL-3_2-11B | Llama-VL-3_2-90B | Llama-3.2V-11B-cot"
)

args = parser.parse_args()

# ------------------- Resolve Model Path -------------------
if "cot" in args.model_name:
    model_path = os.path.join("Xkev", "Llama-3.2V-11B-cot")
else:
    model_path = os.path.join(models_dir, args.model_name)

print(f"Loading model: {args.model_name}")

# ------------------- Output File Path -------------------
result_file_path = os.path.join(Result_root, f"{args.model_name}.json")

# ------------------- Load Model & Processor -------------------
model = MllamaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(model_path)

print(f"Model {args.model_name} loaded successfully.")

# ------------------- Inference Function -------------------
def run_llama(prompt_text, image_path):
    """
    Run LLaMA-VL model on given image and prompt text.
    """

    # Format input as a chat-style conversation
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Apply chat template to generate full input prompt
    input_text = processor.apply_chat_template(
        messages, add_generation_prompt=True
    )

    # Load image
    image = Image.open(image_path)

    # Prepare model inputs
    inputs = processor(image, input_text, return_tensors="pt").to(model.device)

    # Generate output
    output = model.generate(**inputs, max_new_tokens=2048)

    # Decode and return the model's textual response
    decoded_text = processor.decode(output[0])

    # Remove metadata if present (e.g., |end_header_id| separator)
    return decoded_text.split("|end_header_id|")[-1]

# ------------------- Execute Script -------------------
if __name__ == "__main__":
    run_model(root_dir, result_file_path, run_llama)
