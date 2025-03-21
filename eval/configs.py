import os

Real_data_root = "save_real_output/"
Synthetic_root = "save_synthetic_output/"

root_dir = "../"
models_dir = "/data/wengtengjin/models/"
#control the Synthetic Scenario or Real-world Scenario
index = 0
file_dir = ["Synthetic Mathematical Dataset","Real-World Dataset"]
json_path_dir = "../Only_data/" + file_dir[index]

if index == 0:
    Result_root = Synthetic_root
else:
    Result_root = Real_data_root

if not os.path.exists(Result_root):
    os.mkdir(Result_root)

model_list = [
    "Random.json",
    "phi3_5.json",
    "llava-v1.5-7b.json",
    "llava-v1.5-13b.json",
    "llava-v1.6-34b.json",
    "llava-onevision-qwen2-7b-ov-hf.json",
    "llava-onevision-qwen2-72b-si-hf.json",
    "Internvl2_5-8B.json",
    "Internvl2_5-38B.json",
    "Internvl2_5-78B.json",
    "Janus-Pro-7B.json",
    "Qwen2-VL-2B-Instruct.json",
    "Qwen2-VL-7B-Instruct.json",
    "Qwen2-VL-72B-Instruct.json",
    "Qwen2.5-VL-3B-Instruct.json",
    "Qwen2.5-VL-7B-Instruct.json",
    "Qwen2.5-VL-72B-Instruct.json",
    "GPT-4o.json",
    "gemini-1.5-flash.json",
    "gemini-2.0-flash.json",
    "gemini-1.5-pro.json",
    "Human.json",
    "Internvl-8B.json",
    "Internvl-40B.json",
    "InternVL2-8B-MPO.json",
    "llava-v1.5-13b.json",
    "Math-llava-13b.json",
    "Llama-VL-3_2-11B.json",
    "Llama-3.2V-11B-cot.json",
    "Qwen2.5-VL-7B-Instruct.json",
    "R1-Onevision-7B.json"
]