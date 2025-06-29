from pathlib import Path
import json
import os
from configs import json_path_dir

# Construct prompt string for language model
def build_prompt(question, options):
    prompt = f"""
        Question: {question}
        Options: {options}
        Answer the question based on the most likely options.
    """
    return prompt

# Recursively find all JSON files under the specified directory
def get_json_path():
    json_files = [str(file) for file in Path(json_path_dir).rglob('*.json')]
    return json_files

# Retrieve the correct answer for a given image ID
def get_correct_answer(image_id, json_data):
    for item in json_data:
        if item["id"] == image_id:
            return item["answer"]
    return None

# Save results to a JSON file (append if exists)
def write_json(save_path, new_data):
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            existing_data = json.load(f)
        if isinstance(existing_data, list):
            existing_data.append(new_data)
        else:
            existing_data = [existing_data, new_data]
    else:
        existing_data = [new_data]

    with open(save_path, 'w') as f:
        json.dump(existing_data, f, indent=4)

# Main loop: run the model on each sample and save predictions
def run_model(root_dir, save_path, model_run_fn):
    """
    Parameters:
        root_dir: base directory where image files are stored
        save_path: path to save result JSON
        model_run_fn: callable(model_prompt, image_file_path) -> prediction
    """
    json_files = get_json_path()
    for json_file in json_files:
        with open(json_file, 'r') as f:
            dataset = json.load(f)  # Each JSON contains a list of dicts with question data

        for sample in dataset:
            image_id = sample['id']
            question = sample['question']
            image_rel_path = sample["image_path"]
            image_file = os.path.join(root_dir, image_rel_path)
            options = sample["option"]
            correct_answer = sample['answer']

            prompt = build_prompt(question, options)
            predicted_answer = model_run_fn(prompt, image_file)

            result_record = {
                "image_id": image_id,
                "predict_answer": predicted_answer,
                "answer": correct_answer
            }

            write_json(save_path, result_record)

# Compare predicted answer with ground truth (match first 3 characters)
def judge(predicted, correct):
    return predicted == correct[:3]

# Calculate accuracy per category and overall average
def get_accuracy_example(statistics_dict, file_name):
    result_summary = {'model_name': file_name[:-5]}  # strip ".json"
    total_correct = 0
    total_count = 0

    for category, stats in statistics_dict.items():
        correct = stats['correct']
        total = stats['total']
        accuracy = (correct / total) * 100 if total != 0 else 0.0
        result_summary[category] = f"{accuracy:.2f}"

        total_correct += correct
        total_count += total

    average_accuracy = (total_correct / total_count) * 100 if total_count != 0 else 0.0
    result_summary['Ave'] = f"{average_accuracy:.2f}"

    return result_summary
