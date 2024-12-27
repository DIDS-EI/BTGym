from btgym.dataclass.cfg import cfg


"""
# 1. 在其他进程中启动simulator server
# ```shell
# python btgym/simulator/launch_simulator_server.py
# ```
"""

"""
# 2. 在其他进程中启动molmo server
# ```shell
# python btgym/molmo/launch_molmo_server.py
# ```
"""

"""
# 3. 调用LLM生成任务

from btgym.llm.llm import LLM
from btgym.llm.generate_bddl import generate_bddl

llm = LLM()

# 如果想随机选择场景，可以取消注释以下代码
# import random
# with open(f"{cfg.ASSETS_PATH}/scene_list.txt", "r") as f:
#     scene_list = [line.strip() for line in f.readlines()]
# scene_name = random.choice(scene_list)

scene_name = "Rs_int"

bddl = generate_bddl(llm, scene_name)
# 按照命名规则保存bddl （暂未实现）
"""




# # 4. 在仿真器中采样任务
# import json
# from btgym.simulator.simulator_client import SimulatorClient
# simulator_client = SimulatorClient()

# json_path = ''
# while json_path is None:
#     json_path = simulator_client.call(func='SampleCustomTask',
#                                        task_name=cfg.task_name,
#                                        scene_name=cfg.scene_name)


# 5. 在仿真器中读取任务

# from btgym.simulator.simulator_client import SimulatorClient
# simulator_client = SimulatorClient()

# simulator_client.call(func='LoadCustomTask', 
# task_name=cfg.task_name,scene_file_name=cfg.scene_file_name)





# X. 开始执行任务

# from btgym.simulator.simulator_client import SimulatorClient
# 仿真器客户端还在调试






# X. molmo 标点


from btgym.molmo.molmo_client import MolmoClient
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

molmo_client = MolmoClient()

query = f'point out the red pen.'
image_path = f'{cfg.ROOT_PATH}/molmo/camera_0_rgb.png'

generated_text = molmo_client.call(func='PointQA',
                #    query=f'Point out the important parts for doing the task. The task is "reorient the white pen and drop it upright into the black pen holder".',
                   query=query,
                   image_path=image_path
                ).text

image = Image.open(image_path)
points = molmo_client.extract_points(generated_text, image)
draw_points_on_image(image, points, f'{cfg.ROOT_PATH}/molmo/camera_0_rgb_points.png')




"""
# xxx. 保存成功数据
"""