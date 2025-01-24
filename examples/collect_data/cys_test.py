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

cfg.task_name='setting_up_room_for_games'
cfg.scene_name='Rs_int'
cfg.scene_file_name='scene_file_0'
cfg.target_object_name = 'cabinet.n.01_1'

hdf5_path = Path(__file__).parent / 'robot_data.hdf5'


if __name__ == '__main__':
    # sample_task()
    # restart_simulator()
    
    simulator = Simulator()

    json_path = simulator.load_behavior_task(task_name=cfg.task_name)

    for i in range(10):
        simulator.navigate_to_object(object_name=cfg.target_object_name)
        simulator.idle_step(10)
        obs = simulator.get_obs()
        camera_info = simulator.get_camera_info()

        rgb_img = Image.fromarray(obs['rgb'])
        rgb_img.save(f'{CURRENT_DIR}/camera_0_rgb.png')

        molmo_client = MolmoClient()
        # query = f'point out the grasp point of the {cfg.target_object_name.split(".")[0]}. make sure the grasp point is in a stable position and safe.'
        query = f'To open xxx, mark the key points of the handle for opening xxx..'
        # {cfg.target_object_name.split(".")[0]}. make sure the grasp point is in a stable position and safe.'
        point = molmo_client.get_grasp_pose_by_molmo(query,CURRENT_DIR)
        if not point: continue

        target_pos = og_utils.pixel_to_world(obs, camera_info, point[0], point[1])
        target_euler = [math.pi/2, 0, 0]
        target_local_pose = simulator.pose_to_local([*target_pos, *target_euler])

        success = simulator.open_object_by_pose_by_sticky(target_local_pose,object_name=cfg.target_object_name)
        if not success: continue

        obs['gripper_open'] = False
        obs['eef_pose'] = target_local_pose
        add_hdf5_sample(hdf5_path,obs)
        break
    simulator.idle()

