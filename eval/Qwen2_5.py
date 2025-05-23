from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
from utils import run_model
from configs import Result_root, root_dir,models_dir
import argparse

"""
/data/wengtengjin/models/Qwen2-VL-7B-Instruct
/data/wengtengjin/models/Qwen2-VL-2B-Instruct
/data/wengtengjin/models/Qwen2-VL-72B-Instruct
"""
"""
Qwen2.5-VL-3B-Instruct
Qwen2.5-VL-7B-Instruct
Qwen2.5-VL-72B-Instruct
R1-Onevision-7B
"""

parser = argparse.ArgumentParser(description="Script for processing data")

# Add an argument for Data_number
parser.add_argument(
    "--model_name", 
    type=str, 
    default="R1-Onevision-7B", 
    help="Fancy-MLLM/R1-Onevision-7B"
)

# Parse the arguments
args = parser.parse_args()

if "Qwen2.5" not in args.model_name:
    model_path = models_dir + args.model_name
else:
    model_path = "Qwen/" + args.model_name
if args.model_name == "R1-Onevision-7B":
    model_path="Fancy-MLLM/R1-Onevision-7B"
    
file_path = os.path.join(Result_root, model_path.split("/")[-1]+".json")


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)
print(f"{args.model_name} laoded")


def runQwen2_5(text, image_path):
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
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

if __name__ == "__main__":
    run_model(root_dir, file_path, runQwen2_5)