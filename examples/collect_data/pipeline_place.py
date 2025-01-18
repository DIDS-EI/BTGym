from btgym.dataclass import cfg, state

import os
import shutil

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

from btgym.simulator.simulator import Simulator
from btgym.simulator.simulator_client import SimulatorClient

from btgym.molmo.molmo_client import MolmoClient
from PIL import Image, ImageDraw
import json

import h5py
import numpy as np
from pathlib import Path
import time
import math
from btgym.utils.hdf5 import add_hdf5_sample

from btgym.utils import og_utils

cfg.task_name='task1'
cfg.scene_file_name='scene_file_0'
cfg.target_object_name = 'apple.n.01_1'
current_dir = Path(__file__).parent
cfg.hdf5_path = current_dir / 'robot_data.hdf5'

"""
# xxx. 保存成功数据
"""

# 拿起苹果放到桌子上


if __name__ == '__main__':
    # sample_task()
    # restart_simulator()
    
    simulator = Simulator(headless=False)

    simulator.load_custom_task(task_name=cfg.task_name, scene_file_name=cfg.scene_file_name)

    for i in range(10):
        simulator.navigate_to_object(object_name=cfg.target_object_name)
        obs = simulator.get_obs()
        obs['gripper_open'] = False
        obs['eef_pose'] = state.target_local_pose

        grasp_pos = simulator.get_object_pos_by_pose(cfg.target_object_name)['pos'].tolist()
        # simulator.set_target_visual_pose([*grasp_pos,0,0,0])
        
        state.target_local_pose = simulator.pose_to_local(grasp_pos+state.target_euler)
        print(f"local pose: {state.target_local_pose}")
        success = simulator.grasp_object_by_pose(state.target_local_pose,object_name=cfg.target_object_name)
        if success:
            # print(f"第{i}次尝试成功,保存数据")
            add_hdf5_sample(cfg.hdf5_path,obs)
            # break
            
            place_success = False
            for j in range(10):
                cfg.place_target_object = 'coffee_table.n.01_1'
                # 把物体放到桌子上
                simulator.navigate_to_object(object_name=cfg.place_target_object)
                simulator.idle_step(10)
                
                # 在桌子上打点
                obs = simulator.get_obs()
                camera_info = simulator.get_camera_info()
                rgb_img = Image.fromarray(obs['rgb'])
                rgb_img.save(f'{CURRENT_DIR}/camera_0_rgb.png')
                molmo_client = MolmoClient()
                query = f'To place an object on the {cfg.target_object_name.split(".")[0]}, please mark the positions on the {cfg.target_object_name.split(".")[0]} where it can be placed, ensuring the position is stable and safe.'
                point = molmo_client.get_grasp_pose_by_molmo(query,CURRENT_DIR)
                if not point: continue
                
                # point 转为机器人相对坐标
                target_pos = og_utils.pixel_to_world(obs, camera_info, point[0], point[1])
                # 抬高 5厘米 放下
                target_pos[2] += 0.05
                target_euler = [0, math.pi/2, math.pi/2]
                target_local_pose = simulator.pose_to_local([*target_pos, *target_euler])
                
                success = simulator.place_object_by_pose(target_local_pose,object_name=cfg.target_object_name)
                if not success: continue
                place_success = True
                print(f"第{j}次尝试 放置物体成功,保存数据")
                obs['gripper_open'] = True
                obs['eef_pose'] = target_local_pose
                add_hdf5_sample(cfg.hdf5_path,obs)
                break
            
            if not place_success: continue
            break
            
        print(f"第{i}次尝试失败")

    simulator.close()

    
    # 保存数据

    # 保存图像
    # 保存点
    # 保存成功与否
