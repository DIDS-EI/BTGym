import os
import sys
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{DIR}/..')

from examples.scenes.simulator_client import SimulatorClient
from examples.scenes.molmo_client import MolmoClient
from PIL import Image, ImageDraw


def draw_points_on_image(image, points, output_path):
    # 创建图片副本以免修改原图
    img_with_points = image.copy()
    
    # 转换为可绘制格式
    draw = ImageDraw.Draw(img_with_points)
    
    radius = 10
    # 为每个点画一个红色圆圈和序号
    for i, point in enumerate(points):
        x, y = point
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill='green')
        draw.text((x-3, y-6), str(i+1), fill='white', font=None)
    # 保存并显示结果
    img_with_points.convert('RGB').save(output_path)
    # img_with_points.save(output_path)


image_path = f'{DIR}/outputs/image.png'
# image_path = f'{DIR}/outputs/camera_0_rgb.png'
# image_path = f'{DIR}/outputs/camera.jpg'

# simulator_client = SimulatorClient()
# # simulator_client.call(func='LoadTask', task_name='putting_shoes_on_rack')
# object_list = simulator_client.call(func='GetTaskObjects').object_names
# print(object_list)
# object_name = object_list[0]

# # simulator_client.call(func='NavigateToObject', object_name=object_name)
# # response = simulator_client.call(func='SaveCameraImage', output_path=image_path)
query = f'point out the pen.'
# query = f'point out the {object_name.split(".")[0]}.'
# print(query)

molmo_client = MolmoClient()
generated_text = molmo_client.call(func='PointQA',
                #    query=f'Point out the important parts for doing the task. The task is "reorient the white pen and drop it upright into the black pen holder".',
                   query=query,
                   image_path=image_path
                ).text

image = Image.open(image_path)
points = molmo_client.extract_points(generated_text, image)
draw_points_on_image(image, points, f'{DIR}/outputs/camera_with_points.jpg')

# Point out the important parts for doing the task. The task is '{query}'
