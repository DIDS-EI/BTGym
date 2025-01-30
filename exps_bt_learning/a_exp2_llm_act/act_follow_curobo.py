from btgym.dataclass import cfg, state
import os
import shutil
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
import csv
import pandas as pd
from btgym.utils.hdf5 import add_hdf5_sample
import btgym.utils.og_utils as og_utils
import transforms3d.quaternions as T
import transforms3d.euler as E
import torch as th
from btgym.utils.og_utils import direction_vector_to_euler_angles
DIR = Path(__file__).parent
folder_path = os.path.join(DIR.parent, "tasks")
cfg.hdf5_path = DIR.parent.parent / 'examples/collect_data/robot_data.hdf5'
import torch
import omnigibson.utils.transform_utils as T



# cfg.task_name='aaa_demo0_draw3'
cfg.task_name='aaa_demo0_draw4'
cfg.scene_file_name='scene_file_0'

cfg.target_object_name = 'chocolate_cake.n.01_1'




simulator = Simulator(headless=False)
simulator.load_custom_task(task_name=cfg.task_name, scene_file_name=cfg.scene_file_name,folder_path=folder_path)

simulator.set_target_visual_pose([0.0, 0.0, 0.0],size=0.02)
simulator.navigate_to_object(object_name=cfg.target_object_name)
# simulator.idle_step(10)
n = 0
while True:
    cube_position, cube_orientation = simulator.target_visual.get_world_pose()
    # 将四元数转换为欧拉角
    cube_orientation_euler = T.quat2euler(torch.tensor(cube_orientation))

    # reach_pos = th.tensor([*cube_position, *cube_orientation_euler]).clone().detach()
    reach_pos = th.tensor([*cube_position, state.target_euler[0], state.target_euler[1], state.target_euler[2]]).clone().detach()
    try:
        simulator.reach_pose(reach_pos,is_local=False)
        print(reach_pos)
    except Exception as e:
        
        n += 1
        if n > 10:
            # simulator.reset_hand()
            n = 0
        print(f"Error: {e}")
    simulator.idle_step(10)

