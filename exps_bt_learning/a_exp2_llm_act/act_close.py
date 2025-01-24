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



cfg.task_name='task_close'
cfg.scene_file_name='scene_file_0'
cfg.target_object_name = 'cabinet.n.01_1'

   

# calculate the success rate
success_rate = 0
total_try = 10
try_time = 0

# record the data to csv
# 列名: 尝试次数, point 是否正确, curobo 是否成功, 成功与否
data = []

simulator = Simulator(headless=False)
while try_time<=total_try:

    
    simulator.load_custom_task(task_name=cfg.task_name, scene_file_name=cfg.scene_file_name,folder_path=folder_path)
        
    simulator.navigate_to_object(object_name=cfg.target_object_name)
    target_pos = th.tensor([1.5, -8.7688,  0.6])
    obj_face_tensor = simulator.get_object_face_tensor(object_name=cfg.target_object_name,pos=target_pos,horizontal=True)
    yaw = math.atan2(-obj_face_tensor[1],obj_face_tensor[0])
    target_pos = target_pos+obj_face_tensor*0.1
    target_euler = [math.pi/2, 0, yaw]
    # 可视化目标点
    simulator.set_target_visual_pose([*target_pos,0,0,0],size=0.02)
    simulator.idle_step(10)
    target_local_pose = simulator.pose_to_local([*target_pos, *target_euler])
    success = simulator.reach_pose(target_local_pose,is_local=True)
    if success:
        # 先往外推0.5m
        simulator.move_hand_linearly(dir=-obj_face_tensor,distance=0.5)
        simulator.close_gripper()
        # 再向内拉0.3m
        simulator.move_hand_linearly(dir=obj_face_tensor,distance=0.3,ignore_obj_in_hand=True)
        simulator.open_gripper()
        
        while try_time<=total_try:
            try_time+=1
            
            ###################################### Close the drawer #############################################
            simulator.navigate_to_object(object_name=cfg.target_object_name)
            # # # ========= Point by Molmo =========
            obs = simulator.get_obs()
            obs['gripper_open'] = False
            obs['eef_pose'] = state.target_local_pose
            rgb_img = Image.fromarray(obs['rgb'])
            rgb_img.save(f'{DIR}/camera_close_rgb.png')
            molmo_client = MolmoClient()
            query = f'To close the drawer of {cfg.target_object_name.split(".")[0]},'\
                    + f'mark the key points of the handle for opening the drawer of {cfg.target_object_name.split(".")[0]}.'\
                    + f'Identify suitable positions in the upper half of the image to grasp the handle.'
            point = molmo_client.get_grasp_pose_by_molmo(query,dir=DIR,point_img_path=f'{DIR}/camera_close_rgb.png')
            # # =========              ========= 
            if not point:
                print(f"第{try_time}次尝试失败, point 为空")
                data.append([try_time, False, False, False])
                continue
            
            # try_time+=1
            # point 转为世界坐标
            camera_info = simulator.get_camera_info()
            target_pos = og_utils.pixel_to_world(obs, camera_info, point[0], point[1])
            # target_pos = th.tensor([1.5, -8.7688,  0.6])
            
            
            simulator.navigate_to_pos(object_name=cfg.target_object_name,pos=target_pos,offset=(0.9,-0.2))
            simulator.open_gripper()
            # point 转为机器人相对坐标
            camera_info = simulator.get_camera_info()
            target_pos = og_utils.pixel_to_world(obs, camera_info, point[0], point[1]) #tensor([ 1.7293, -8.5120,  0.6488], dtype=torch.float64)
            
            simulator.navigate_to_pos(object_name=cfg.target_object_name,pos=target_pos,offset=(0.9,-0.2))
            
            # 获取朝向
            obj_face_tensor = simulator.get_object_face_tensor(object_name=cfg.target_object_name,pos=target_pos,horizontal=True)
            yaw = math.atan2(-obj_face_tensor[1],obj_face_tensor[0])

            target_pos = target_pos+obj_face_tensor*0.1
            target_euler = [math.pi/2, 0, yaw]

            
            target_local_pose = simulator.pose_to_local([*target_pos, *target_euler])
            success = simulator.reach_pose(target_local_pose,is_local=True)
            
            if not success:continue
            
            # 先往外推0.5m
            simulator.move_hand_linearly(dir=-obj_face_tensor,distance=0.5)
            
            
            # 用 BDDL 判断是否成功
            goal_list = simulator.og_sim.task.ground_goal_state_options[0]
            for goal in goal_list:
                print("goal.terms:",goal.terms) #['ontop', 'printer.n.03_1', 'table.n.02_1']
                print("goal.currently_satisfied:",goal.currently_satisfied)
            if goal_list[0].currently_satisfied:                  
                print(f"第{try_time}次尝试成功,保存数据")
                add_hdf5_sample(cfg.hdf5_path,obs)
                success_rate += 1
                # data.append([try_time, point, True, True])
                data.append([try_time, target_pos, True, True])
                break
            ###################################### Close the drawer #############################################
        
        print(f"第{try_time}次尝试失败")
    
    
    
    simulator.reset()


print(f"{cfg.task_name} SR: {success_rate}/{total_try}")
print(f"{cfg.task_name} SR: {success_rate/total_try}*100%")

# save the data to csv
# 用 pd.DataFrame 保存
df = pd.DataFrame(data, columns=['try_num', 'point_correct', 'curobo_success', 'success'])
df.to_csv(f'{DIR}/../results/exp2_{cfg.task_name}_open_{total_try}.csv', index=False)
    
simulator.close()
