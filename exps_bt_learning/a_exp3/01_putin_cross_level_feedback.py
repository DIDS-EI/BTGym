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



cfg.task_name='aaa_demo1_putin_fail'
cfg.scene_file_name='scene_file_0'
cfg.target_object_name = 'cabinet.n.01_1'


simulator = Simulator(headless=False)
simulator.load_custom_task(task_name=cfg.task_name, scene_file_name=cfg.scene_file_name,folder_path=folder_path,\
    load_task_relevant_only=True)


# simulator.idle()

def open(object_name,query):

    simulator.navigate_to_object(object_name=object_name)

    # target_pos_word = simulator.keypoint_proposal(query)

    simulator.open_gripper()
    
    target_pos_word = th.tensor([ 0.45527, -3.50259,  0.64625])
    simulator.set_target_visual_pose(target_pos_word,size=0.05)
    simulator.idle_step(10)
    simulator.navigate_to_pos(object_name=cfg.target_object_name,pos=target_pos_word,offset=(0.9,-0.2))
    
    simulator.eef_reach_pos(target_pos_word,horizontal=True)
    
    simulator.move_hand_forward(0.3)
    
    simulator.close_gripper()
    
    simulator.move_hand_backward(0.3)

    simulator.open_gripper()
        

def grasp(object_name,query):
    
    simulator.navigate_to_object(object_name=object_name)
    
    target_pos_word = simulator.keypoint_proposal(query)
    
    simulator.open_gripper()
    
    simulator.eef_reach_pos(target_pos_word,horizontal=False)
    
    simulator.close_gripper()
    
    
def placein(object_name,query):
    
    simulator.navigate_to_object(object_name=object_name)
    
    # simulator.idle()
    
    target_pos_word = simulator.keypoint_proposal(query)
    
    simulator.eef_reach_pos(target_pos_word,horizontal=False)
    
    simulator.open_gripper()
    
    
    

def grasp_grounding(object_name):
    simulator.navigate_to_object(object_name=object_name)
    target_pos_word = simulator.get_object_pos_by_pose(object_name)['pos']
    
    # simulator.idle()
    
    # 可视化这个点
    simulator.set_target_visual_pose([*target_pos_word,0,0,0],size=0.02)
    simulator.idle_step(10)
    
    simulator.open_gripper()
    simulator.eef_reach_pos(target_pos_word,horizontal=False,grounding=True)
    simulator.close_gripper()

if __name__ == "__main__":
    
    # # 拿起物体
    # cfg.target_object_name = 'apple.n.01_1'
    # grasp_grounding(cfg.target_object_name)
    
    # # 放入抽屉
    # cfg.target_object_name = 'cabinet.n.01_1'
    # query = f'To place an object on the {cfg.target_object_name.split(".")[0]}, '\
    #         + f'please identify suitable positions in the upper half of the image where it can be placed. '\
    #         + f'Ensure that the selected position is stable and safe.'
    # placein(cfg.target_object_name,query)
    


    
    # # 打开抽屉
    cfg.target_object_name = 'cabinet.n.01_1'
    query = f'To open the drawer of {cfg.target_object_name.split(".")[0]},'\
            + f'mark the key points of the handle for opening the drawer of {cfg.target_object_name.split(".")[0]}.'\
            + f'Identify suitable positions in the upper half of the image to grasp the handle.'
    open(cfg.target_object_name,query)
    # # 用 BDDL 判断是否成功
    # goal_list = simulator.og_sim.task.ground_goal_state_options[0]
    # for goal in goal_list:
    #     print("goal.terms:",goal.terms) 
    #     print("goal.currently_satisfied:",goal.currently_satisfied)


    simulator.close()
        
        