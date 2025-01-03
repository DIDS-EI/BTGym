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


def parse_point(center,point, w_h):
    """
    将二分法坐标转换为图片上的相对坐标(0-1之间的小数)
    point: [(x1,y1), (x2,y2), (x3,y3), ...]
    return: (x, y) 相对坐标
    """
    print(point)
    x, y = 0.0, 0.0
    w, h = w_h
    for i, (px, py) in enumerate(point):
        x += px * (0.5 ** (i+1))
        y += py * (0.5 ** (i+1))
    y = 1 - y
    return center[0]-w/2 + x*w, center[1]-h/2 + y*h
    

def draw_points(img, point_dict):
    with open(os.path.join(folder_path, "camera_0_bbox_center_dict.pkl"), "rb") as f:
        point_center_dict = pickle.load(f)

    print(point_center_dict)
    # h, w = img.shape[:2]
    print(point_dict)
    for object_name in point_dict:
        for name, point in point_dict[object_name].items():
            center_point, w_h = point_center_dict[object_name]
            x, y = parse_point(center_point, point, w_h)
            point_pos = (int(x), int(y))    
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
    instruction = "put the red pen on the other free space of the table"
    with open(os.path.join(folder_path, "camera_0_bbox_center_dict.pkl"), "rb") as f:
        point_center_dict = pickle.load(f)

    object_list = list(point_center_dict.keys())

    cg = ConstraintGenerator()
    # output = cg.run_vlm(image_path=os.path.join(folder_path, "camera_0_merged_objects.png"),
    output = cg.run_vlm(image_path=os.path.join(folder_path, "camera_0_bbox_2d.png"),
    # output = cg.run_vlm(image_path=os.path.join(folder_path, "camera_0_seg_labeled.png"),
                instruction="""
你是一个智能机器人，现在需要根据人类指令来完成任务.你应该按照以下过程来一步步聪明地思考:

<thinking_process>
1. 相关物体分析:先在图片中找到任务相关物体和他们的序号，如3_pen, 4_pen holder
    具体任务会提供一个物体列表，从中选取相关物体名即可
2. 关键点提议:对每个物体输出语义上有意义的关键点名称。对于每个物体，可以将关键点分为以下三类，：
    a. 抓取关键点，需要成对出现，用来描述两个夹爪的抓取位置
    b. 目标关键点，通常用来描述物体的移动相对目标，例如笔的中心点相对于笔架的中心点
    c. 姿态关键点，通常用来描述物体移动或最终的姿态，例如笔需要竖直放置
3. 关键点标注：指出关键点在图片中的位置，这个位置是对应相关物体的外框的，选关键点的过程采用二分法思维链：
    使用[(x_1,y_1),(x_2,y_2),...]的坐标对表示关键点，其中x_1,y_1表示第一个坐标对，x_2,y_2表示第二个坐标对，
    逐渐对图片的外框进行细分，最终确定关键点相对于图片外框的位置。
    其中x坐标只能为left或者right，y坐标只能为up或者bottom。不允许出现center等其他词.
4. 任务执行代码输出：
    - keypoint_dict: 储存每个物体的每个关键点信息
    - do_task(env)函数：编写代码来完成指令提到的任务，最后会提供一些env可用的方法。
</thinking_process>

<example>
你的当前任务是：
把红色的笔放到笔筒中

1. 相关物体分析:
根据对指令的分析，得到相关物体为3_pen, 4_pen holder

2. 关键点选取:
对于笔来说，需要描述的关键点有：
- 抓取关键点，pen_grasp_point
- 目标关键点，pen_target_point_center
- 姿态关键点，已经有二个关键点，足以复用关键点来描述姿态
对于笔筒来说，需要描述的关键点有：
- 目标关键点，pencil_holder_target_point_bottom_center

3. 关键点标注：
pen_grasp_point:
由于笔是长形物体，应该从笔的侧边进行抓取。为了保证抓取的稳定性，可考虑抓取笔的中心，该位置可以这样描述：
首先考虑笔的左下区域(left,bottom)，然后进一步选择该区域的右上区域(right,up)，最后进一步再选择右上区域(right,up)。 
由此得到 pen_grasp_point = [(left,bottom),(right,up),(right,up)]

(其他点的标注过程类似，不再赘述)

4. 基于类的代码输出：
```python
keypoint_dict = {
    "3_pen": {
        "pen_grasp_point": [(left,bottom),(right,up),(right,up)],
        "pen_target_point_center": [(left,bottom),(left,up),(left,up)],
    },
    "4_pencil_holder": {
        "pencil_holder_target_point_center": [(left,bottom),(right,up),(right,up)],
    }
}

def do_task(env):
    # 首先抓取笔
    env.grasp_pos(keypoint_dict["3_pen"]["grasp_point_pen"])
    # 然后移动到笔筒上方
    pen_release_pose = pencil_holder.get_pen_release_pose(pen)
    env.reach_pose(pen_release_pose)
    # 最后释放笔
    env.release_object()

```
</example>

"""+\
f"""
你现在的任务是:
`{instruction}`

相关任务信息有：
1. 所有物体列表: {object_list}
2. env对象可用方法：
    - env.grasp_pos(keypoint)： 从关键点抓取对应的物体
    - env.reach_pose(): 到达目标位姿，如果手上有物体则保持抓取
    - env.release_object(): 松开手中的物体
以下是你的关键点思考过程和最终结果输出：
""")

    left = 0
    right = 1
    bottom = 0
    up = 1
    point_dict = {}
    code = extract_code(output)
    exec(code)


    # point_dict = {
    #     "3_pen": {
    #         "grasp_point_pen": [(right,up),(left,bottom),(left,bottom)],
    #         "target_point_pen_center": [(right,bottom),(right,up),(right,up)],
    #     },
    #     "4_pencil_holder": {
    #         "target_point_pencil_holder_center": [(right,up),(left,bottom),(left,bottom)],
    #     }
    # }

    img = cv2.imread(os.path.join(folder_path, "camera_0_rgb.png"))
    merged_img = draw_points(img, point_dict)
    # merged_img = merge_imgs_with_points(img_list, point_dict_list)
    cv2.imwrite(os.path.join(folder_path, "camera_0_merged_objects_with_points.png"), merged_img)
    print(point_dict)
