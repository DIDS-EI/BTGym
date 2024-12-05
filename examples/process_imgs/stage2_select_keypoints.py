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
import pickle
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


def merge_imgs_with_lines(img_list):
    """
    img_list: list of cv2 images
    """
    merged_width = 360
    margin = 20
    processed_imgs = []
    for img in img_list:
        h, w = img.shape[:2]
        # 如果高度大于宽度，旋转90度
        if h > w:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            h, w = w, h
        # 等比缩放到目标宽度
        scale = merged_width / w
        new_h = int(h * scale)
        img = cv2.resize(img, (merged_width, new_h))
        
        h, w = img.shape[:2]
        # 画2x2的粗线
        y_mid = h // 2
        x_mid = w // 2
        # 横向粗线
        cv2.line(img, (0, y_mid), (w, y_mid), (0, 255, 0), 4)
        # 纵向粗线
        cv2.line(img, (x_mid, 0), (x_mid, h), (0, 255, 0), 4)
        
        # 画4x4的细线
        # 横向细线
        y1 = h // 4
        y2 = h * 3 // 4
        cv2.line(img, (0, y1), (w, y1), (0, 255, 0), 1)
        cv2.line(img, (0, y2), (w, y2), (0, 255, 0), 1)
        
        # 纵向细线
        x1 = w // 4
        x2 = w * 3 // 4
        cv2.line(img, (x1, 0), (x1, h), (0, 255, 0), 1)
        cv2.line(img, (x2, 0), (x2, h), (0, 255, 0), 1)
        
        processed_imgs.append(img)

    # 计算总高度并创建画布
    total_height = sum([img.shape[0] for img in processed_imgs]) + margin * (len(processed_imgs) - 1)
    merged_img = np.zeros((total_height, merged_width, 3), dtype=np.uint8)
    
    # 依次粘贴图片
    y_offset = 0
    for img in processed_imgs:
        h = img.shape[0]
        merged_img[y_offset:y_offset+h, :] = img
        y_offset += h + margin

    
    return merged_img



def merge_imgs(img_list):
    """
    img_list: list of cv2 images
    """
    merged_width = 360
    margin = 20
    processed_imgs = []
    for img in img_list:
        h, w = img.shape[:2]
        # 如果高度大于宽度，旋转90度
        if h > w:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            h, w = w, h
        # 等比缩放到目标宽度
        scale = merged_width / w
        new_h = int(h * scale)
        img = cv2.resize(img, (merged_width, new_h))
        
        h, w = img.shape[:2]
        
        processed_imgs.append(img)

    # 计算总高度并创建画布
    total_height = sum([img.shape[0] for img in processed_imgs]) + margin * (len(processed_imgs) - 1)
    merged_img = np.zeros((total_height, merged_width, 3), dtype=np.uint8)
    
    # 依次粘贴图片
    y_offset = 0
    for img in processed_imgs:
        h = img.shape[0]
        merged_img[y_offset:y_offset+h, :] = img
        y_offset += h + margin

    
    return merged_img


def parse_point(center,point):
    """
    将二分法坐标转换为图片上的相对坐标(0-1之间的小数)
    point: [(x1,y1), (x2,y2), (x3,y3), ...]
    return: (x, y) 相对坐标
    """
    print(point)
    x, y = center[0], center[1]
    for i, (px, py) in enumerate(point):
        x += px * (0.5 ** (i+1))
        y += py * (0.5 ** (i+1))
    y = 1 - y
    return x, y
    

def draw_points(img, point_dict):
    with open(os.path.join(folder_path, "camera_0_bbox_center_dict.pkl"), "rb") as f:
        point_center_dict = pickle.load(f)

    h, w = img.shape[:2]

    for object_name in point_dict:
        for name, point in point_dict[object_name].items():
            x, y = parse_point(point_center_dict[object_name], point)
            point_pos = (int(x*w), int(y*h))
            cv2.circle(img, point_pos, 5, (255, 255, 255), -1)
            cv2.putText(img, name, (point_pos[0]+10, point_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img

def merge_imgs_with_points(img_list, point_dict_list):
    merged_width = 360
    margin = 20
    processed_imgs = []
    for i, img in enumerate(img_list):
        h, w = img.shape[:2]
        # 如果高度大于宽度，旋转90度
        if h > w:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            h, w = w, h
        # 等比缩放到目标宽度
        scale = merged_width / w
        new_h = int(h * scale)
        img = cv2.resize(img, (merged_width, new_h))
        
        h, w = img.shape[:2]
        for name, point in point_dict_list[i].items():
            x, y = parse_point(point)
            point_pos = (int(x*w), int(y*h))
            cv2.circle(img, point_pos, 5, (255, 255, 255), -1)
            cv2.putText(img, name, (point_pos[0]+10, point_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        processed_imgs.append(img)

    # 计算总高度并创建画布
    total_height = sum([img.shape[0] for img in processed_imgs]) + margin * (len(processed_imgs) - 1)
    merged_img = np.zeros((total_height, merged_width, 3), dtype=np.uint8)
    
    # 依次粘贴图片
    y_offset = 0
    for img in processed_imgs:
        h = img.shape[0]
        merged_img[y_offset:y_offset+h, :] = img
        y_offset += h + margin

    
    return merged_img

def extract_code(string):
    # 使用正则表达式提取代码
    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, string, re.DOTALL)
    if match:
        answer = match.group(1).strip()
    else:
        raise ValueError("No code found in the answer")
    return answer





if __name__ == "__main__":
    # img_name_list = [ "camera_0_object_2.png", "camera_0_object_3.png" ]
    # img_list = []
    # for img_name in img_name_list:
    #     img_path = os.path.join(folder_path, img_name)
    #     img = cv2.imread(img_path)
    #     img_list.append(img)

    # merged_img = merge_imgs(img_list)
    # merged_img = merge_imgs_with_lines(img_list)

    # save_path = os.path.join(folder_path, "camera_0_merged_objects.png")
    # cv2.imwrite(save_path, merged_img)


    cg = ConstraintGenerator()
    # output = cg.run_vlm(image_path=os.path.join(folder_path, "camera_0_merged_objects.png"),
    output = cg.run_vlm(image_path=os.path.join(folder_path, "camera_0_bbox_2d.png"),
    # output = cg.run_vlm(image_path=os.path.join(folder_path, "camera_0_seg_labeled.png"),
                instruction="""
the current task is:
`reorient the red pen and drop it upright into the black pen holder`
首先你需要在图片中找到相关物体和他们的序号，如3_pen, 4_pen holder


你现在需要在图片中给每个相关物体标注关键点，选关键点的过程采用二分法思维链。

关键点分类：
1. 抓取关键点，需要成对出现，用来描述两个夹爪的抓取位置
2. 目标关键点，通常用来描述物体的移动相对目标，例如笔的中心点相对于笔架的中心点
3. 姿态关键点，通常用来描述物体移动或最终的姿态，例如笔需要竖直放置

整体的思考过程分为两步：
1. 先思考对于每个物体，需要哪些关键点，例如：
对于笔来说，需要描述的关键点有：
- 抓取关键点，grasp_point
- 目标关键点，target_point_pen_center
- 姿态关键点，已经有二个关键点，足以复用关键点来描述姿态
对于笔筒来说，需要描述的关键点有：
- 目标关键点，target_point_pencil_holder_center

2. 确定每个关键点的坐标，使用二分法思维链：
使用[(x_1,y_1),(x_2,y_2),...]的坐标对表示关键点，其中x_1,y_1表示第一个坐标对，x_2,y_2表示第二个坐标对，以此类推。
其中x坐标只能为left或者right，y坐标只能为up或者bottom。不允许出现center!!!

思考过程如下所示：
笔的关键点确定：
- 抓取关键点
由于笔是长形物体，应该从笔的侧边进行抓取。在图片上两个抓取点应该是上下关系,且尽量位于物体中心。
对于grasp_point_pen来说：
1. 首先对图片四等分，选取笔的左下区域，相应坐标为(left,bottom)
2. 然后再进一步四等分，选取以上区域的右上区域，相应坐标为(right,up)
3. 最后再进一步四等分，选取以上区域的右上角点作为笔的中心点，相应坐标为(right,up)
最后得到 grasp_point_pen = [(left,bottom),(right,up),(right,up)]

对于target_point_pen_center来说：
1. 首先对图片四等分，选取笔的左下区域，相应坐标为(left,bottom)
2. 然后再进一步四等分，选取以上区域的左上区域，相应坐标为(left,up)
3. 最后再进一步四等分，选取以上区域的左上角点作为笔的中心点，相应坐标为(left,up)
最后得到 target_point_pen_center = [(left,bottom),(left,up),(left,up)]

对于target_point_pencil_holder_center来说：
- 目标关键点
由于笔筒是圆柱形物体，应该从笔筒的侧边进行抓取。在图片上两个抓取点应该是上下关系,且尽量位于物体中心。
1. 首先对图片四等分，选取笔筒的左下区域，相应坐标为(left,bottom)
2. 然后再进一步四等分，选取以上区域的右上区域，相应坐标为(right,up)
3. 最后再进一步四等分，选取以上区域的右上角点作为笔筒的中心点，相应坐标为(right,up)
最后得到 pencil_holder_center_point = [(left,bottom),(right,up),(right,up)]


最终结果：
```python
point_dict = {
    "3_pen": {
        "grasp_point_pen": [(left,bottom),(right,up),(right,up)],
        "target_point_pen_center": [(left,bottom),(left,up),(left,up)],
    },
    "4_pen_holder": {
        "target_point_pencil_holder_center": [(left,bottom),(right,up),(right,up)],
    }
}
```

以下是你的关键点思考过程和最终结果输出：
""")
    left = 0
    right = 1
    bottom = 0
    up = 1
    point_dict_list = []
    code = extract_code(output)
    exec(code)

    img = cv2.imread(os.path.join(folder_path, "camera_0_rgb.png"))
    merged_img = draw_points(img, point_dict)
    # merged_img = merge_imgs_with_points(img_list, point_dict_list)
    cv2.imwrite(os.path.join(folder_path, "camera_0_merged_objects_with_points.png"), merged_img)
    print(point_dict_list)
