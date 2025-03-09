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
from openai import OpenAI
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


#
"""
GPT4o
"""
parser = argparse.ArgumentParser(description="Script for processing data")
# Add an argument for Data_number
parser.add_argument(
    "--model_name", 
    type=str, 
    default="GPT-4o", 
    help="GPT-4o"
)
# Parse the arguments
args = parser.parse_args()

model_path = models_dir + args.model_name
file_path = os.path.join(Result_root, model_path.split("/")[-1]+".txt")
api_key = ""

base_url="https://api.61798.cn/v1"
client = OpenAI(api_key=api_key, base_url=base_url)

import time

def runGpt_4v(prompt, image_path):
    print(prompt, image_path)

    # 限制请求速率
    request_limit_per_minute = 10
    request_interval = 60 / request_limit_per_minute
    time.sleep(request_interval)

    # 将本地图片转换为 Base64
    image_base64 = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ],
            }
        ],
        max_tokens=300,
    )

    return response.choices[0].message.content
    
if __name__ == "__main__":
    run_model(root_dir, file_path, runGpt_4v)
    # print(runGpt_4v("1",2))

