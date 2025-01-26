from btgym.dataclass import cfg, state

import os
import shutil

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

from btgym.simulator.simulator_client import SimulatorClient

from btgym.molmo.molmo_client import MolmoClient
from PIL import Image, ImageDraw
import json

import h5py
import numpy as np
from pathlib import Path
import time
import math

cfg.task_name='task1'
cfg.scene_file_name='scene_file_0'
cfg.target_object_name = 'apple.n.01_1'



simulator_client = SimulatorClient()
molmo_client = MolmoClient()

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


def restart_simulator():
    simulator_client.call(func='Close')

# # 4. 在仿真器中采样任务

def sample_task():
    json_path = ''
    while json_path == '':
        json_path = simulator_client.call(func='SampleCustomTask',
                                        task_name=cfg.task_name,
                                        scene_name=cfg.scene_name).json_path
    print('场景json保存在：',json_path)

    # 复制文件并重命名为scene_file_0.json
    target_path = os.path.join(cfg.task_folder,cfg.task_name, f'{cfg.scene_file_name}.json')
    shutil.copy2(json_path, target_path)
    print('已复制场景文件到:', target_path)


# 5. 在仿真器中读取任务
def load_task():
    response = simulator_client.call(func='LoadCustomTask', task_name=cfg.task_name, scene_file_name=cfg.scene_file_name)



# 6. 导航到物体，获取图像
def get_image():
    simulator_client.call(func='NavigateToObject', object_name=cfg.target_object_name)
    object_pos = simulator_client.call(func='GetObjectPos', object_name=cfg.target_object_name).pos
    print('object_pos',object_pos)
    # simulator_client.call(func='SetCameraLookatPos', pos=object_pos)

    # 获取图像
    response = simulator_client.get_obs()
    response = simulator_client.get_camera_info()

    rgb = simulator_client.obs['rgb']
    rgb_img = Image.fromarray(rgb[:,:,:3])

    rgb_img.save(f'{CURRENT_DIR}/camera_0_rgb.png')


# 7. molmo 在图像中标点

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

def get_grasp_pose_by_molmo():
    query = f'point out the grasp point of the {cfg.target_object_name.split(".")[0]}. make sure the grasp point is in a stable position and safe.'

    print('query:',query)
    image_path = f'{CURRENT_DIR}/camera_0_rgb.png'

    generated_text = molmo_client.call(func='PointQA',
                    #    query=f'Point out the important parts for doing the task. The task is "reorient the white pen and drop it upright into the black pen holder".',
                    query=query,
                    image_path=image_path
                    ).text

    image = Image.open(image_path)
    points = molmo_client.extract_points(generated_text, image)
    print('molmo points',points)
    draw_points_on_image(image, points, f'{CURRENT_DIR}/camera_0_rgb_points.png')


    # 图像上的点转换到世界坐标
    if len(points) > 0:
        state.target_pos = simulator_client.pixel_to_world(int(points[0][0]),int(points[0][1])).cpu().numpy()
        response = simulator_client.call(func='SetTargetVisualPose', pose=[*state.target_pos, 0, 0, 0])

        state.target_local_pose = simulator_client.call(func='PoseToLocal', pose=[*state.target_pos, *state.target_euler]).pose
        return state.target_local_pose
    else:
        print('molmo没有标出任何点！')
        return None
    # response = client.call(func='SetCameraLookatPos', pos=pos)


def get_grasp_pose_by_object():

    state.target_pos = simulator_client.call(func='GetObjectPos', object_name=cfg.target_object_name).pos
    state.target_local_pose = simulator_client.call(func='PoseToLocal', pose=[*state.target_pos, *state.target_euler]).pose
    return state.target_local_pose



# 9. 根据点来抓取物体
def grasp_object(grasp_pose):
    # simulator_client.call(func='NavigateToObject', object_name=cfg.target_object_name)
    # grasp_pos = simulator_client.call(func='GetObjectPos', object_name=cfg.target_object_name).pos
    # grasp_pos = grasp_pos['pos'].tolist()
    print('开始抓取物体 grasp_pose',grasp_pose)
    success = simulator_client.call(func='GraspObjectByPos', pos=grasp_pose, object_name=cfg.target_object_name).success
    return success



def save_data():
    current_dir = Path(__file__).parent
    hdf5_path = current_dir / 'robot_data.hdf5'

    # Open the existing file in read/write mode
    if not os.path.exists(hdf5_path):
        with h5py.File(hdf5_path, 'w') as f:
            data_group = f.create_group('data')
            f.attrs['total_samples'] = 0

    with h5py.File(hdf5_path, 'r+') as f:
        data_group = f['data']
        current_samples = f.attrs['total_samples']
        total_samples = current_samples + 1

        sample_group = data_group.create_group(f'sample_{total_samples-1:05d}')

        sample_group.create_dataset('rgb', 
                                    data=simulator_client.obs['rgb'],
                                    compression='gzip',
                                    compression_opts=9)
        
        sample_group.create_dataset('depth', 
                                    data=simulator_client.obs['depth'],
                                    compression='gzip',
                                    compression_opts=9)
        
        sample_group.create_dataset('seg_semantic', 
                                    data=simulator_client.obs['seg_semantic'],
                                    compression='gzip',
                                    compression_opts=9)
        
        sample_group.create_dataset('proprio', 
                                    data=simulator_client.obs['proprio'])
        
        sample_group.create_dataset('gripper_open', 
                                    data=False)
        
        sample_group.create_dataset('eef_pose', 
                                    data=state.target_local_pose)

        f.attrs['total_samples'] = total_samples

"""
# xxx. 保存成功数据
"""


if __name__ == '__main__':
    sample_task()
    restart_simulator()
    load_task()

    for i in range(10):
        get_image()
        grasp_pose = get_grasp_pose_by_object()
        

        if grasp_pose is not None:
            success = grasp_object(grasp_pose)
            if success:
                save_data()
                break

    
    # 保存数据

    # 保存图像
    # 保存点
    # 保存成功与否
