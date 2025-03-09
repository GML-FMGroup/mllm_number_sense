import os
Synthetic_root = "Synthetic_results/"
Real_data_root = "Real_results"
root_dir = "/data/wengtengjin/wtj_works/"
models_dir = "/data/wengtengjin/models/"
index = 0
file_dir = ["Synthetic Mathematical Dataset","Real-World Dataset"]
json_path_dir = "/data/wengtengjin/wtj_works/Only_data/" + file_dir[index]
# print(json_path_dir)
if index == 0:
    Result_root = Synthetic_root
else:
    Result_root = Real_data_root

if not os.path.exists(Result_root):
    os.mkdir(Result_root)


model_list = [
    "Random.txt",
    "phi3_5.txt",
    "llava-v1.5-7b.txt",
    "llava-v1.5-13b.txt",
    "llava-v1.6-34b.txt",
    "llava-onevision-qwen2-7b-ov-hf.txt",
    "llava-onevision-qwen2-72b-si-hf.txt",
    "Internvl2_5-8B.txt",
    "Internvl2_5-38B.txt",
    "Internvl2_5-78B.txt",
    "Janus-Pro-7B.txt",
    "Qwen2-VL-2B-Instruct.txt",
    "Qwen2-VL-7B-Instruct.txt",
    "Qwen2-VL-72B-Instruct.txt",
    "Qwen2.5-VL-3B-Instruct.txt",
    "Qwen2.5-VL-7B-Instruct.txt",
    "Qwen2.5-VL-72B-Instruct.txt",
    "GPT-4o.txt",
    "gemini-1.5-flash.txt",
    "gemini-2.0-flash.txt",
    "gemini-1.5-pro.txt",
    "Human.txt",
    "Internvl-8B.txt",
    "Internvl-40B.txt",
    "InternVL2-8B-MPO.txt",
    "llava-v1.5-13b.txt",
    "Math-llava-13b.txt",
    "Llama-VL-3_2-11B.txt",
    "Llama-3.2V-11B-cot.txt",
    "Qwen2.5-VL-7B-Instruct.txt",
    "R1-Onevision-7B.txt"
]

COTandLVM_list = [
    "Internvl-8B.txt",
    # "InternVL2-8B-MPO.txt",
    # "llava-v1.5-13b.txt",
    "Math-llava-13b.txt",
    # "Llama-VL-3_2-11B.txt",
    # "Llama-VL-3_2-11B.txt",
    # "Qwen2.5-VL-7B-Instruct.txt",
    # "R1-Onevision-7B",
]