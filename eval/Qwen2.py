import os
import argparse
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils import run_model
from configs import Result_root, root_dir, models_dir

"""
Available model paths:
    /data/wengtengjin/models/Qwen2-VL-2B-Instruct
    /data/wengtengjin/models/Qwen2-VL-7B-Instruct
    /data/wengtengjin/models/Qwen2-VL-72B-Instruct
"""

# ------------------- Argument Parsing -------------------
parser = argparse.ArgumentParser(description="Run Qwen2-VL model on dataset")

parser.add_argument(
    "--model_name",
    type=str,
    default="Qwen2-VL-2B-Instruct",
    help="Specify the model name: Qwen2-VL-2B-Instruct, Qwen2-VL-7B-Instruct, Qwen2-VL-72B-Instruct, etc."
)

args = parser.parse_args()

model_path = os.path.join(models_dir, args.model_name)

# Output result file path
result_file_path = os.path.join(Result_root, f"{args.model_name}.json")

print(f"Loading model: {args.model_name} from {model_path}")

# ------------------- Load Model & Processor -------------------
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

print(f"Model {args.model_name} loaded successfully.")

# ------------------- Model Inference Wrapper -------------------
def run_qwen(prompt_text, image_path):
    """
    Given a prompt and an image path, generate a model response.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text}
            ],
        }
    ]

    # Prepare input prompt and media
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    # Tokenize all inputs
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # Generate prediction
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    trimmed_ids = [
        output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, generated_ids)
    ]

    decoded_output = processor.batch_decode(
        trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return decoded_output[0]

# ------------------- Run -------------------
if __name__ == "__main__":
    run_model(root_dir, result_file_path, run_qwen)
