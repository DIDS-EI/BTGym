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
DIR = Path(__file__).parent



# 传入图片和相机信息，返回全局坐标系下的目标点
def keypoint_proposal(simulator,query,img_dir=DIR,file_name='camera_0_rgb.png'):
    
    obs = simulator.get_obs()
    rgb_img = Image.fromarray(obs['rgb'])
    camera_info = simulator.get_camera_info()

    
    rgb_img = Image.fromarray(obs['rgb'])
    
    rgb_img.save(f'{img_dir}/{file_name}')
    molmo_client = MolmoClient()
    point = molmo_client.get_grasp_pose_by_molmo(query,dir=img_dir,point_img_path=f'{img_dir}/{file_name}')
    if point:
        target_pos = og_utils.pixel_to_world(obs, camera_info, point[0], point[1])
        simulator.navigate_to_pos(object_name=cfg.target_object_name,pos=target_pos,offset=(0.9,-0.2))
    else:
        target_pos = None
    return target_pos


# 用于水平抓取获取 抓取点
def eef_reach_pos(simulator,pos,horizontal=True):
    target_pos = pos
    
    if horizontal:
        obj_face_tensor = th.tensor([0,1,0.])
        yaw = math.atan2(-obj_face_tensor[1],obj_face_tensor[0])

        target_pos = target_pos+obj_face_tensor*0.1
        target_euler = [math.pi/2, 0, yaw]

        # 可视化目标点
        simulator.set_target_visual_pose([*target_pos,0,0,0],size=0.02)
        simulator.idle_step(10)

        target_local_pose = simulator.pose_to_local([*target_pos, *target_euler])
        success = simulator.reach_pose(target_local_pose,is_local=True)
    else:
        pass

def move_hand_forward(simulator,distance=0.5):
    obj_face_tensor=th.tensor([0,1,0.])
    simulator.move_hand_linearly(dir=-obj_face_tensor,distance=distance)


def move_hand_backward(simulator,distance=0.3):
    obj_face_tensor=th.tensor([0,1,0.])
    simulator.move_hand_linearly(dir=obj_face_tensor,distance=distance,ignore_obj_in_hand=True)