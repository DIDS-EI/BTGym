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
import torch as th

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

    target_pos = th.tensor([1.5, -8.7688,  0.62])
    simulator.navigate_to_pos(object_name=cfg.target_object_name,pos=target_pos,offset=(0.9,-0.2))

    simulator.open_gripper()

    obj_face_tensor = simulator.get_object_face_tensor(object_name=cfg.target_object_name,pos=target_pos,horizontal=True)
    yaw = math.atan2(-obj_face_tensor[1],obj_face_tensor[0])

    target_pos = target_pos+obj_face_tensor*0.1
    target_euler = [math.pi/2, 0, yaw]

    simulator.set_target_visual_pose([*target_pos,0,0,0],size=0.02)
    simulator.idle_step(10)
    target_local_pose = simulator.pose_to_local([*target_pos, *target_euler])

    success = simulator.reach_pose(target_local_pose,is_local=True)

    # 先往外推0.5m
    simulator.move_hand_linearly(dir=-obj_face_tensor,distance=0.5)
    simulator.close_gripper()

    # 再向内拉0.3m
    simulator.move_hand_linearly(dir=obj_face_tensor,distance=0.3,ignore_obj_in_hand=True)

    simulator.open_gripper()
    
    
    # close
    for i in range(10):
        simulator.navigate_to_object(object_name=cfg.target_object_name)
        simulator.open_gripper()
        
        # ========= 打个点 有点效果
        obs = simulator.get_obs()
        obs['gripper_open'] = False
        obs['eef_pose'] = state.target_local_pose
            
        rgb_img = Image.fromarray(obs['rgb'])
        rgb_img.save(f'{CURRENT_DIR}/camera_grasp_rgb.png')
        molmo_client = MolmoClient()
        # 
        # query = f'point out the handle of the {cfg.target_object_name.split(".")[0]} to push and close.'
        # query = f'要关闭 {cfg.target_object_name.split(".")[0]},标出  {cfg.target_object_name.split(".")[0]} 的把手位置'
        query = f'There is an open {cfg.target_object_name.split(".")[0]} in the image, mark the handle position of {cfg.target_object_name.split(".")[0]}, it must be marked'
        # query = f'There is an open {cfg.target_object_name.split(".")[0]} in the image, mark the operable position, such as the handle, to facilitate closing it'
        # query = f'point out the position to place objects'
        point = molmo_client.get_grasp_pose_by_molmo(query,CURRENT_DIR,point_img_path=f'{CURRENT_DIR}/camera_grasp_rgb.png')
        # ========= 
        
        if not point: 
            continue
        
        # point 转为机器人相对坐标
        camera_info = simulator.get_camera_info()
        target_pos = og_utils.pixel_to_world(obs, camera_info, point[0], point[1]) #tensor([ 1.7293, -8.5120,  0.6488], dtype=torch.float64)
        
        # simulator.navigate_to_pos(object_name=cfg.target_object_name,pos=target_pos,offset=(0.9,-0.2))
        
        # 获取朝向
        obj_face_tensor = simulator.get_object_face_tensor(object_name=cfg.target_object_name,pos=target_pos,horizontal=True)
        yaw = math.atan2(-obj_face_tensor[1],obj_face_tensor[0])

        target_pos = target_pos+obj_face_tensor*0.1
        target_euler = [math.pi/2, 0, yaw]

        
        target_local_pose = simulator.pose_to_local([*target_pos, *target_euler])
        success = simulator.reach_pose(target_local_pose,is_local=True)
        
        if not success:
            continue
        
        # 先往外推0.5m
        simulator.move_hand_linearly(dir=-obj_face_tensor,distance=0.5)
        
        if success:
            break
        
    
    
    # simulator.close_gripper()
    # # 再向内拉0.3m
    # simulator.move_hand_linearly(dir=-obj_face_tensor,distance=0.3,ignore_obj_in_hand=True)
    
    # for i in range(10):
    #     simulator.navigate_to_object(object_name=cfg.target_object_name)
    #     simulator.idle_step(10)
    #     obs = simulator.get_obs()
    #     camera_info = simulator.get_camera_info()

    #     rgb_img = Image.fromarray(obs['rgb'])
    #     rgb_img.save(f'{CURRENT_DIR}/camera_0_rgb.png')

    #     molmo_client = MolmoClient()
    #     query = f'point out the grasp point of the {cfg.target_object_name.split(".")[0]}. make sure the grasp point is in a stable position and safe.'
    #     point = molmo_client.get_grasp_pose_by_molmo(query,CURRENT_DIR)
    #     if not point: continue

    #     target_pos = og_utils.pixel_to_world(obs, camera_info, point[0], point[1])
    #     print(f"target world pos: {target_pos}")
    #     target_euler = [math.pi/2, 0, 0]
    #     target_local_pose = simulator.pose_to_local([*target_pos, *target_euler])

    #     success = simulator.grasp_object_by_pose(target_local_pose,object_name=cfg.target_object_name)
    #     if not success: continue

    #     obs['gripper_open'] = False
    #     obs['eef_pose'] = target_local_pose
    #     add_hdf5_sample(hdf5_path,obs)
    #     break
    simulator.idle()

