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



cfg.task_name='task1'
cfg.scene_file_name='scene_file_0'

cfg.target_object_name = 'apple.n.01_1'
cfg.target_place_name = 'coffee_table.n.01_1'



# if __name__ == '__main__':
    

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


    while try_time<=total_try:
        
        simulator.navigate_to_object(object_name=cfg.target_object_name)
        grasp_pos = simulator.get_object_pos_by_pose(cfg.target_object_name)['pos'].tolist()
        state.target_local_pose = simulator.pose_to_local(grasp_pos+state.target_euler)
        print(f"local pose: {state.target_local_pose}")
        success = simulator.grasp_object_by_pose(state.target_local_pose,object_name=cfg.target_object_name)
        if success:
            
            ######################## Place ########################
            place_success = False
            while try_time<=total_try:
                try_time+=1
                
                simulator.navigate_to_object(object_name=cfg.target_place_name)
                simulator.idle_step(10)
                
                # 在桌子上打点
                obs = simulator.get_obs()
                camera_info = simulator.get_camera_info()
                rgb_img = Image.fromarray(obs['rgb'])
                rgb_img.save(f'{DIR}/camera_place_rgb.png')
                molmo_client = MolmoClient()
                
                query = f'To place an object on the {cfg.target_object_name.split(".")[0]}, '\
                        + f'please identify suitable positions in the upper half of the image where it can be placed. '\
                        + f'Ensure that the selected position is stable and safe.'
                        # +f'Do not point to the robot base in the lower half of the image.'
                        
                point = molmo_client.get_grasp_pose_by_molmo(query,DIR,point_img_path=f'{DIR}/camera_place_rgb.png')
                if not point: 
                    print(f"第{try_time}次尝试失败, point 为空")
                    data.append([try_time, False, False, False])
                    continue
                
                target_pos = og_utils.pixel_to_world(obs, camera_info, point[0], point[1])
                # 抬高 5厘米 放下
                target_pos[2] += 0.05
                target_euler = [0, math.pi/2, math.pi/2]
                target_local_pose = simulator.pose_to_local([*target_pos, *target_euler])
                
                success = simulator.place_object_by_pose(target_local_pose,object_name=cfg.target_object_name)
                if success: 
                    
                    # 用 bddl 判断是否成功
                    goal_list = simulator.og_sim.task.ground_goal_state_options[0]
                    for goal in goal_list:
                        print("goal.terms:",goal.terms) #['ontop', 'printer.n.03_1', 'table.n.02_1']
                        print("goal.currently_satisfied:",goal.currently_satisfied)
                    if goal_list[0].currently_satisfied:                  
                        print(f"第{try_time}次尝试成功, 放置物体成功")
                        data.append([try_time, point, True, True])
                        success_rate += 1
                        place_success = True
                        break
                    else:
                        print(f"第{try_time}次尝试成功, 放置物体失败")
                        data.append([try_time, point, False, False])
                        break
                else:
                    print(f"第{try_time}次尝试成功, 放置物体失败")
                    data.append([try_time, point, False, False])
                    continue
            ######################## Place ########################   
    simulator.reset()


print(f"{cfg.task_name} SR: {success_rate}/{total_try}")
print(f"{cfg.task_name} SR: {success_rate/total_try}*100%")

# save the data to csv
# 用 pd.DataFrame 保存
df = pd.DataFrame(data, columns=['try_num', 'point_correct', 'curobo_success', 'success'])
df.to_csv(f'{DIR}/../results/exp2_{cfg.task_name}_place_{total_try}.csv', index=False)
    
simulator.close()
