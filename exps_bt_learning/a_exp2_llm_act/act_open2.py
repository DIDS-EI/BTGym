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
# from exps_bt_learning.a_exp2_llm_act.action_funs import keypoint_proposal,eef_reach_pos,move_hand_forward,move_hand_backward


DIR = Path(__file__).parent
folder_path = os.path.join(DIR.parent, "tasks")
cfg.hdf5_path = DIR.parent.parent / 'examples/collect_data/robot_data.hdf5'



cfg.task_name='task2'
cfg.scene_file_name='scene_file_0'
cfg.target_object_name = 'cabinet.n.01_1'


simulator = Simulator(headless=False)
simulator.load_custom_task(task_name=cfg.task_name, scene_file_name=cfg.scene_file_name,folder_path=folder_path)


def open(query):

    simulator.navigate_to_object(object_name=cfg.target_object_name)

    target_pos_word = simulator.keypoint_proposal(query)

    simulator.open_gripper()
    
    target_pos_word = th.tensor([ 1.6412, -8.8041,  0.62544])
    simulator.eef_reach_pos(target_pos_word,horizontal=True)
    
    simulator.move_hand_forward(0.5)
    
    simulator.close_gripper()
    
    simulator.move_hand_backward(0.5)

    simulator.open_gripper()
        





if __name__ == "__main__":
    
    
    query = f'To open the drawer of {cfg.target_object_name.split(".")[0]},'\
            + f'mark the key points of the handle for opening the drawer of {cfg.target_object_name.split(".")[0]}.'\
            + f'Identify suitable positions in the upper half of the image to grasp the handle.'
    
    open(query)
    
    # 用 BDDL 判断是否成功
    goal_list = simulator.og_sim.task.ground_goal_state_options[0]
    for goal in goal_list:
        print("goal.terms:",goal.terms) 
        print("goal.currently_satisfied:",goal.currently_satisfied)


    simulator.close()
        
        