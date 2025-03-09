import os
from utils import run_model
from configs import Result_root, root_dir, models_dir
import argparse
import pathlib
import textwrap
import PIL.Image
import google.generativeai as genai
import time
import argparse
#
"""
gemini-1.5-flash
gemini-1.5-pro
"""
parser = argparse.ArgumentParser(description="Script for processing data")
# Add an argument for Data_number
parser.add_argument(
    "--model_name", 
    type=str, 
    default="gemini-1.5-pro", 
    help="gemini-1.5-pro;gemini-1.5-flash;gemini-2.0-flash"
)
# Parse the arguments
args = parser.parse_args()

model_path = models_dir + args.model_name
file_path = os.path.join(Result_root, model_path.split("/")[-1]+".txt")
# print(file_path)


# def to_markdown(text):
#     text = text.replace("•", "  *")
#     return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))
keys = [
]
index = 0

def Gemini(prompt, image_file):
    global index  # 需要用 global 来修改 index
    print(prompt)
    # 配置 API key
    genai.configure(api_key=keys[index])
    model = genai.GenerativeModel(args.model_name)

    # 根据模型类型设置请求间隔时间
    if args.model_name == "gemini-1.5-flash" or args.model_name == "gemini-2.0-flash":
        time.sleep(5)  # 每 5 秒尝试一次请求（每分钟 15 个请求）
    else:
        time.sleep(32)

    img = PIL.Image.open(image_file)

    try:
        # 尝试生成内容
        response = model.generate_content(
            [
                prompt,
                img,
            ],
            stream=True,
        )
        response.resolve()
        # print(response.text)
        return response.text
    except Exception as e:
        # 如果发生错误（如超出API key额度），切换到下一个key
        print(f"Error occurred: {e}. Switching to next API key.")
        
        # 增加 index，切换到下一个 key
        index += 1
        
        # 检查是否超出所有 key 的范围
        if index >= len(keys):
            print("All API keys have been used up.")
            return None  # 如果所有的 key 都用完了，返回 None 或根据需要返回其他值
        return Gemini(prompt, image_file)  # 递归调用 Gemini 使用下一个 key
    
if __name__ == "__main__":
    run_model(root_dir, file_path, Gemini)
        

