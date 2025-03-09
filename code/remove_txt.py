import os

# file_dir = "Synthetic_results"
file_dir = "Real_results"

# 需要过滤的单词列表
# filter_words = ["Depth","Quantity"]

# 遍历目录中的每个文本文件
for txt_file in os.listdir(file_dir):
    # 构造文件的完整路径
    file_path = os.path.join(file_dir, txt_file)
    
    # 打开文件并读取内容
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # 过滤掉以指定单词开头的行
    filtered_lines = [line for line in lines if not any(line.startswith(word) for word in filter_words)]
    
    # 将过滤后的内容写回到文件
    with open(file_path, 'w') as file:
        file.writelines(filtered_lines)