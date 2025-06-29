import os
import argparse
import torch
from transformers import AutoModelForCausalLM

from utils import run_model
from configs import Result_root, root_dir, models_dir
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

"""
Example model path:
- /data/wengtengjin/models/llava-onevision-qwen2-72b-si-hf
- deepseek-ai/Janus-Pro-7B
"""

# ------------------- Argument Parsing -------------------
parser = argparse.ArgumentParser(description="Run Janus Multi-Modal Model")

parser.add_argument(
    "--model_name",
    type=str,
    default="deepseek-ai/Janus-Pro-7B",
    help="Model name, e.g., deepseek-ai/Janus-Pro-7B or local dir"
)

args = parser.parse_args()
model_path = args.model_name  # allow HuggingFace or local path

print(f"Loading model: {model_path}")

# ------------------- Output File Path -------------------
result_file_path = os.path.join(Result_root, model_path.split("/")[-1] + ".json")

# ------------------- Load Processor & Model -------------------
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

# Load Janus multi-modality language model
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

print(f"Model {args.model_name} loaded successfully.")

# ------------------- Inference Function -------------------
def run_janus(prompt_text, image_path):
    """
    Run Janus multi-modal model on given image and prompt.
    """

    # Define a conversation following Janus format
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{prompt_text}",
            "images": [image_path],
        },
        {
            "role": "<|Assistant|>",
            "content": ""
        },
    ]

    # Load image and preprocess inputs
    pil_images = load_pil_images(conversation)

    inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
    ).to(vl_gpt.device)

    # Generate embeddings from image+text inputs
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**inputs)

    # Generate output response
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )

    # Decode output tokens into readable text
    response = tokenizer.decode(
        outputs[0].cpu().tolist(),
        skip_special_tokens=True
    )

    return response

# ------------------- Execute Script -------------------
if __name__ == "__main__":
    run_model(root_dir, result_file_path, run_janus)
