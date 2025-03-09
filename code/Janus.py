import os
from utils import run_model
from configs import Result_root, root_dir, models_dir
import argparse
import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

"""
/data/wengtengjin/models/llava-onevision-qwen2-72b-si-hf
"""

parser = argparse.ArgumentParser(description="Script for processing data")

# Add an argument for Data_number
parser.add_argument(
    "--model_name", 
    type=str, 
    default="llava-onevision-qwen2-72b-si-hf", 
    help="llava-onevision-qwen2-72b-si-hf"
)
model_path = "deepseek-ai/Janus-Pro-7B"
# Parse the arguments
args = parser.parse_args()
# model_path = models_dir + args.model_name

print(f"{args.model_name} laoded")
file_path = os.path.join(Result_root, model_path.split("/")[-1]+".txt")
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()


def run_janus(text, image_path):
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{text}",
            "images": [image_path],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)

    # # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # # run the model to get the response
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    # return f"{prepare_inputs['sft_format'][0]}", answer
    return answer


if __name__ == "__main__":
    run_model(root_dir, file_path, run_janus)
