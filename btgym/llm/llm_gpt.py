import base64
import os
import cv2
import openai
from datetime import datetime

class LLM:
    def __init__(self, model='gpt-4o', temperature=0.5, max_tokens=2048):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        openai.api_base = os.getenv('OPENAI_BASE_URL')
        openai.api_key = os.getenv('OPENAI_API_KEY')

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def request_instruction(self, text, image_path=None):  # generate 生成
        messages = [{"role": "user", "content": text}]
        
        if image_path:
            img_base64 = self.encode_image(image_path)
            # 将图像数据嵌入到字符串中
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text
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
        else:
            messages = [{"role": "user", "content": text}]


        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response.choices[0].message.content

if __name__ == "__main__":      
    # 使用示例
    print("llm init...")
    llm = LLM()
    
    # print("llm generate text...")
    # text_response = llm.request_instruction("请给我一个关于人工智能的简短介绍。")
    # print("Text Response:", text_response)

    print("llm generate image...")
    image_path = os.path.join(os.path.dirname(__file__), "../../examples/collect_data/camera_0_rgb.png")
    image_response = llm.request_instruction("请描述这张图片。", image_path=image_path)
    print("Image Response:", image_response)