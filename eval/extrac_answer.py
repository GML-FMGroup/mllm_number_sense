import os
import json
import torch
from transformers import AutoTokenizer, AutoModel
from configs import index, model_list, models_dir

# ------------------- Load Evaluation Model -------------------
# You can replace this with any powerful LLM (e.g., GPT-4 or Gemini Flash 2.0)
model_path = os.path.join(models_dir, "Internvl2_5-38B")
device_map = "balanced"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map=device_map
).eval()

# ------------------- Model Inference -------------------
def run_model_get(prompt):
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    response = model.chat(tokenizer, None, prompt, generation_config)
    return response

# ------------------- Prompt Construction -------------------
def build_prompt(question, options, prediction):
    """
    Builds a prompt for evaluating whether a model's free-form answer
    aligns with one of the structured multiple-choice options.
    """
    template = (
        "You are an AI assistant who will help me to match an answer with several options of a single-choice question. "
        "You are provided with a question, several options, and an answer, and you need to find which option is most similar to the answer. "
        "If the answer says things like refuse to answer, I'm sorry cannot help, etc., output (z). "
        "If the meaning of all options are significantly different from the answer, or the answer does not select any option, output (z). "
        "You should output one of the choices, (a), (b), (c), (d), (e) (if they are valid options), or (z).\n\n"
        "Example 1:\n"
        "Question: Which point is closer to the camera?\nOptions: (a) Point A\n(b) Point B\n(z) Failed\n"
        "Answer: Point B, where the child is sitting, is closer to the camera.\nYour output: (b)\n\n"
        "Example 2:\n"
        "Question: Which point is closer to the camera?\nOptions: (a) Point A\n(b) Point B\n(z) Failed\n"
        "Answer: I'm sorry, but I can't assist with that request.\nYour output: (z)\n\n"
        "Example 3:\n"
        "Question: Which point is corresponding to the reference point?\n"
        "Options: (a) Point A\n(b) Point B\n(c) Point C\n(d) Point D\n(z) Failed\n"
        "Answer: ... (answer that doesn't match any option) ...\nYour output: (z)\n\n"
        "Now, evaluate the following:\n"
        "Question: {}\nOptions: {}\n(z) Failed\nAnswer: {}\nYour output: "
    )
    return template.format(question, options, prediction)

# ------------------- Retrieve Question & Options -------------------
def get_text_option(image_id, json_data):
    for item in json_data:
        if item["id"] == image_id:
            return item["question"], item["option"]
    return None, None

# ------------------- File and Directory Setup -------------------
save_results_dir = ["save_synthetic_output", "save_real_output"]
dataset_path = [
    "../datasets/Synthetic Mathematical Dataset/Synthetic.json",
    "../datasets/Real-World Dataset/Real.json"
]

# Load source questions/options
with open(dataset_path[index], 'r') as f:
    question_option_data = json.load(f)

input_dir = os.path.join(os.getcwd(), save_results_dir[index])
output_dir = input_dir.replace("output", "prediction")

os.makedirs(output_dir, exist_ok=True)

# ------------------- Main Evaluation Loop -------------------
for file_name in model_list:
    input_json_path = os.path.join(input_dir, file_name)
    output_json_path = os.path.join(output_dir, file_name)

    if not os.path.exists(input_json_path):
        continue

    with open(input_json_path, 'r') as f:
        predictions_data = json.load(f)

    results = []
    for item in predictions_data:
        image_id = item["image_id"]
        model_answer = item["predict_answer"]
        question, options = get_text_option(image_id, question_option_data)

        if not question or not options:
            continue  # Skip if data is incomplete

        prompt = build_prompt(question, options, model_answer)
        matched_option = run_model_get(prompt)

        results.append({image_id: matched_option})

    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)
