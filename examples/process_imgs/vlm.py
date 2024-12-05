import base64
from openai import OpenAI
import os
import cv2
import json
import parse
import numpy as np
import time
from datetime import datetime
from btgym.llm.llm import LLM
import re
folder_path = os.path.dirname(os.path.abspath(__file__))

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_code(string):
    # 使用正则表达式提取代码
    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, string, re.DOTALL)
    if match:
        answer = match.group(1).strip()
    else:
        raise ValueError("No code found in the answer")
    return answer


class ConstraintGenerator:
    def __init__(self):
        self.llm = LLM()

    def run_vlm(self, image_path, instruction):
        img_base64 = encode_image(image_path)
        # save prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": instruction
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    },
                ]
            }
        ]
        return self.llm.custom_request(messages)

if __name__ == "__main__":
    cg = ConstraintGenerator()
    cg.run_vlm(image_path=os.path.join(folder_path, "camera_0_rgb_grids.png"),
                instruction="""
需要完成以下任务：
现在要将红色的笔放到笔筒里

按照以下步骤一步步思考：
1. 为每个物体标注一些关键点，每个关键点一定要保留2位小数的精度,例如1.23,2.56等，禁止出现1.00这种不精确的情况。
每个物体上的关键点分为3类。
    - 1.1 抓取关键点，用2个点表示，用来表示两边夹爪的抓取位置，无须代码处理
    - 1.2 目标关键点，用任意数量的点表示，用来描述物体的目标位姿pose，需要编写代码通过这些点得到目标位姿
    - 1.3 约束关键点，用任意数量的点表示，用来描述物体在运动过程中需要保证的姿态约束，需要编写代码得到姿态约束
2. 为每个对象输出python代码，每个对象描述为一个类，如Pen,PenHolder，每个类有get_grasp_pose和get_path_constraints两个函数，
分别对应目标位姿和姿态约束的获取
""")

    