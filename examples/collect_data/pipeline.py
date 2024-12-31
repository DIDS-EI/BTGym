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

cfg.task_name='task1'
cfg.scene_file_name='scene_file_0'
cfg.target_object_name = 'apple.n.01_1'
current_dir = Path(__file__).parent
cfg.hdf5_path = current_dir / 'robot_data.hdf5'

def save_data():
    # Open the existing file in read/write mode
    if not os.path.exists(cfg.hdf5_path):
        with h5py.File(cfg.hdf5_path, 'w') as f:
            data_group = f.create_group('data')
            f.attrs['total_samples'] = 0

    with h5py.File(cfg.hdf5_path, 'r+') as f:
        data_group = f['data']
        current_samples = f.attrs['total_samples']

        sample_group = data_group.create_group(f'{current_samples:08d}')

        sample_group.create_dataset('rgb', 
                                    data=state.rgb,
                                    compression='gzip',
                                    compression_opts=9)
        
        sample_group.create_dataset('depth', 
                                    data=state.depth,
                                    compression='gzip',
                                    compression_opts=9)
        
        sample_group.create_dataset('seg_semantic', 
                                    data=state.seg_semantic,
                                    compression='gzip',
                                    compression_opts=9)
        
        sample_group.create_dataset('proprio', 
                                    data=state.proprio)
        
        sample_group.create_dataset('gripper_open', 
                                    data=state.gripper_open)
        
        sample_group.create_dataset('eef_pose', 
                                    data=state.target_local_pose)

        f.attrs['total_samples'] = current_samples + 1

"""
# xxx. 保存成功数据
"""


if __name__ == '__main__':
    # sample_task()
    # restart_simulator()
    
    simulator = Simulator()

    simulator.load_custom_task(task_name=cfg.task_name, scene_file_name=cfg.scene_file_name)

    for i in range(10):
        simulator.navigate_to_object(object_name=cfg.target_object_name)
        obs = simulator.get_obs()
        state.rgb = obs['rgb']
        state.depth = obs['depth']
        state.seg_semantic = obs['seg_semantic']
        state.proprio = obs['proprio']
        state.gripper_open = False

        grasp_pos = simulator.get_object_pos_by_pose(cfg.target_object_name)['pos'].tolist()
        # simulator.set_target_visual_pose([*grasp_pos,0,0,0])
        state.target_local_pose = simulator.pose_to_local(grasp_pos+state.target_euler)
        print(f"local pose: {state.target_local_pose}")
        success = simulator.grasp_object_by_pose(state.target_local_pose,object_name=cfg.target_object_name)
        if success:
            print(f"第{i}次尝试成功,保存数据")
            save_data()
            break

        print(f"第{i}次尝试失败")

    simulator.close()

    
    # 保存数据

    # 保存图像
    # 保存点
    # 保存成功与否
