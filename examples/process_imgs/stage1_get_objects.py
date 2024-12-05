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

folder_path = os.path.dirname(os.path.abspath(__file__))

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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
    cg.run_vlm(image_path=os.path.join(folder_path, "camera_0_seg_labeled.png"),
                instruction="""
the current task is:
`reorient the red pen and drop it upright into the black pen holder`
please output the object ids of the objects involved in this task
""")
