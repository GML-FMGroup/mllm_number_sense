import os
Synthetic_root = "Synthetic_results/"
Real_data_root = "Real_results"
root_dir = "/data/wengtengjin"
models_dir = "/data/wengtengjin/models/"
index = 0
file_dir = ["Synthetic Mathematical Dataset","Real-World Dataset"]
json_path_dir = "/data/wengtengjin/Only_data/" + file_dir[index]
# print(json_path_dir)
if index == 0:
    Result_root = Synthetic_root
else:
    Result_root = Real_data_root

if not os.path.exists(Result_root):
    os.mkdir(Result_root)