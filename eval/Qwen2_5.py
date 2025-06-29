import os
import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils import run_model
from configs import Result_root, root_dir, models_dir

"""
Example model paths:
- /data/wengtengjin/models/Qwen2.5-VL-3B-Instruct
- /data/wengtengjin/models/Qwen2.5-VL-7B-Instruct
- /data/wengtengjin/models/Qwen2.5-VL-72B-Instruct
- Fancy-MLLM/R1-Onevision-7B
"""

# ------------------- Argument Parsing -------------------
parser = argparse.ArgumentParser(description="Run Qwen2.5-VL or compatible model")

parser.add_argument(
    "--model_name", 
    type=str, 
    default="Qwen2.5-VL-3B-Instruct", 
    help="Specify model name (e.g., Qwen2.5-VL-7B-Instruct or R1-Onevision-7B)"
)

args = parser.parse_args()

# ------------------- Determine Model Path -------------------
if args.model_name == "R1-Onevision-7B":
    model_path = os.path.join("Fancy-MLLM", args.model_name)

model_path = os.path.join("Qwen", args.model_name)

# ------------------- Output File Path -------------------
result_file_path = os.path.join(Result_root, f"{args.model_name}.json")

print(f"Loading model: {args.model_name} from {model_path}")

# ------------------- Load Model & Processor -------------------
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

print(f"Model {args.model_name} loaded successfully.")

# ------------------- Inference Function -------------------
def run_qwen2_5(prompt_text, image_path):
    """
    Run Qwen2.5-VL model on a given image and prompt.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Format prompt using processor's chat template
    formatted_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Preprocess image/video inputs
    image_inputs, video_inputs = process_vision_info(messages)

    # Tokenize inputs
    inputs = processor(
        text=[formatted_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # Generate model output
    generated_ids = model.generate(**inputs, max_new_tokens=4096)

    # Trim inputs to isolate new tokens
    trimmed_ids = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]

    # Decode generated tokens
    output_text = processor.batch_decode(
        trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]

# ------------------- Execute Main Pipeline -------------------
if __name__ == "__main__":
    run_model(root_dir, result_file_path, run_qwen2_5)
