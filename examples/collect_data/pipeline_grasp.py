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


cfg.task_name='task1'
cfg.scene_file_name='scene_file_0'
cfg.target_object_name = 'apple.n.01_1'
current_dir = Path(__file__).parent
cfg.hdf5_path = current_dir / 'robot_data.hdf5'

"""
# xxx. 保存成功数据
"""


if __name__ == '__main__':
    # sample_task()
    # restart_simulator()
    
    simulator = Simulator(headless=False)

    simulator.load_custom_task(task_name=cfg.task_name, scene_file_name=cfg.scene_file_name, load_task_relevant_only=False)

    for i in range(10):
        simulator.navigate_to_object(object_name=cfg.target_object_name)
        obs = simulator.get_obs()
        obs['gripper_open'] = False
        obs['eef_pose'] = state.target_local_pose
        
        # ========= 打个点 有点效果
        rgb_img = Image.fromarray(obs['rgb'])
        rgb_img.save(f'{CURRENT_DIR}/camera_grasp_rgb.png')
        molmo_client = MolmoClient()
        query = f'point out the {cfg.target_object_name.split(".")[0]}.'
        point = molmo_client.get_grasp_pose_by_molmo(query,CURRENT_DIR,point_img_path=f'{CURRENT_DIR}/camera_grasp_rgb.png')
        # ========= 

        grasp_pos = simulator.get_object_pos_by_pose(cfg.target_object_name)['pos'].tolist()
        # simulator.set_target_visual_pose([*grasp_pos,0,0,0])
        
        state.target_local_pose = simulator.pose_to_local(grasp_pos+state.target_euler)
        print(f"local pose: {state.target_local_pose}")
        success = simulator.grasp_object_by_pose(state.target_local_pose,object_name=cfg.target_object_name)
        if success:
            print(f"第{i}次尝试成功,保存数据")
            add_hdf5_sample(cfg.hdf5_path,obs)
            break

        print(f"第{i}次尝试失败")

    simulator.close()

    
    # 保存数据

    # 保存图像
    # 保存点
    # 保存成功与否
