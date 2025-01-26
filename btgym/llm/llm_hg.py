import os
from transformers import LlamaForCausalLM, LlamaTokenizer   #, PrismaticVLMProcessor, PrismaticVLMModel
import torch
from PIL import Image
from prismatic import load
import timm


class MultiModalModel:
    def __init__(self, text_model_path='path/to/llama3', vlm_model_path='path/to/prismatic_vlm'):
        # 加载文本模型
        self.text_tokenizer = LlamaTokenizer.from_pretrained(text_model_path)
        self.text_model = LlamaForCausalLM.from_pretrained(text_model_path)

        # 加载视觉语言模型
        # self.vlm_processor = PrismaticVLMProcessor.from_pretrained(vlm_model_path)
        # self.vlm_model = PrismaticVLMModel.from_pretrained(vlm_model_path)
        # 直接导入已经下载好的模型
        
        model = timm.create_model( # 加载模型时指定本地路径
            'vit_large_patch14_reg4_dinov2',
            pretrained=False,
            checkpoint_path='/home/admin01/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/2718d189d269d0bf6a0d62ed7aa920cdc688ec84/model.safetensors'
        )
        # 加载 prism-dinosiglip+7b
        model_path = "/home/admin01/workspace/Hugging_Face/prism-dinosiglip+7b"
        self.vlm_model = load(model_path)
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vlm_model.to(device, dtype=torch.bfloat16)

        

    def generate_text(self, prompt, max_length=2048, temperature=0.5):
        # 编码输入文本
        inputs = self.text_tokenizer(prompt, return_tensors="pt")
        
        # 生成文本
        outputs = self.text_model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature
        )
        
        # 解码输出文本
        return self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def process_image(self, image_path, text_prompt):
        # 加载并处理图像
        image = Image.open(image_path)
        inputs = self.vlm_processor(text=text_prompt, images=image, return_tensors="pt")
        
        # 获取视觉语言模型的输出
        outputs = self.vlm_model(**inputs)
        
        # 处理输出（假设输出是一个文本描述）
        return outputs.text




if __name__ == "__main__":
    model = MultiModalModel(
        text_model_path='huggingface/llama3',  # 替换为 Hugging Face 上的模型路径
        vlm_model_path='huggingface/prismatic_vlm'  # 替换为 Hugging Face 上的模型路径
    )
    
    # 文本生成示例
    text_response = model.generate_text("请给我一个关于人工智能的简短介绍。")
    print("Text Response:", text_response)
    
    # 图像处理示例
    print("llm generate image...")
    image_path = os.path.join(os.path.dirname(__file__), "../../examples/collect_data/camera_0_rgb.png")
    image_response = model.process_image(image_path, "请描述这张图片。")
    print("Image Response:", image_response)