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



# cfg.task_name='aaa_demo0_draw3'
cfg.task_name='aaa_demo0_draw4'
# cfg.task_name='aaa_demo0_draw3_garden'
cfg.scene_file_name='scene_file_0'

cfg.target_object_name = 'chocolate_cake.n.01_1'
# cfg.target_object_name = 'apple.n.01_1'



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
        try_time+=1
        

        simulator.navigate_to_object(object_name=cfg.target_object_name)
        simulator.idle_step(10)
        simulator.open_gripper()
        grasp_pos = simulator.get_object_pos_by_pose(cfg.target_object_name)['pos'].tolist()
        # grasp_pos[2] = grasp_pos[2] + 0.05
        # grasp_pos[1] = grasp_pos[1] + 0.15
        # grasp_pos[0] = grasp_pos[0] - 0.15
        
        
        # 可视化这个点
        simulator.set_target_visual_pose([*grasp_pos,0,0,0],size=0.02)
        simulator.idle_step(10)
        
        # state.target_euler = [0, -math.pi/2, math.pi/2]
        
        # state.target_local_pose = simulator.pose_to_local(grasp_pos+state.target_euler)
        # print(f"local pose: {state.target_local_pose}")
        success = simulator.grasp_object_by_pose(grasp_pos+state.target_euler,\
            object_name=cfg.target_object_name,is_local=False)
        if success:
            print(f"第{try_time}次尝试成功,保存数据")
            add_hdf5_sample(cfg.hdf5_path,obs)
            success_rate += 1
            data.append([try_time, point, True, True])
            break
        else:
            data.append([try_time, point, False, False])
        print(f"第{try_time}次尝试失败")
    
    
    
    simulator.reset()


print(f"{cfg.task_name} SR: {success_rate}/{total_try}")
print(f"{cfg.task_name} SR: {success_rate/total_try}*100%")

# save the data to csv
# 用 pd.DataFrame 保存
df = pd.DataFrame(data, columns=['try_num', 'point_correct', 'curobo_success', 'success'])
df.to_csv(f'{DIR}/../results/exp2_{cfg.task_name}_grasp_{total_try}.csv', index=False)
    
simulator.close()
