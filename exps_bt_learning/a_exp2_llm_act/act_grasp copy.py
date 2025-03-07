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




# if __name__ == '__main__':
    

# calculate the success rate
success_rate = 0
total_try = 10
try_time = 0

# record the data to csv
# 列名: 尝试次数, point 是否正确, curobo 是否成功, 成功与否
data = []


while try_time<=total_try:

    simulator = Simulator(headless=False)
    simulator.load_custom_task(task_name=cfg.task_name, scene_file_name=cfg.scene_file_name,folder_path=folder_path)


    while try_time<=total_try:
        
        try_time+=1
        
        simulator.navigate_to_object(object_name=cfg.target_object_name)
        
        # # ========= Point by Molmo =========
        # obs = simulator.get_obs()
        # obs['gripper_open'] = False
        # obs['eef_pose'] = state.target_local_pose
        # rgb_img = Image.fromarray(obs['rgb'])
        
        # rgb_img.save(f'{DIR}/camera_{cfg.task_name}_rgb.png')
        # molmo_client = MolmoClient()
        # query = f'The image contains {cfg.target_object_name.split(".")[0]}. Point out the most likely {cfg.target_object_name.split(".")[0]} in the image.'
        # point = molmo_client.get_grasp_pose_by_molmo(query,dir=DIR,point_img_path=f'{DIR}/camera_{cfg.task_name}_rgb.png')
        # # =========                ========= 
        # if not point:
        #     print(f"第{i}次尝试失败, point 为空")
        #     data.append([i, False, False, False])
        #     continue
        # # point 转为世界坐标
        # camera_info = simulator.get_camera_info()
        # target_pos = og_utils.pixel_to_world(obs, camera_info, point[0], point[1])
        
        # grasp_pos = simulator.get_object_pos_by_pose(cfg.target_object_name)['pos'].tolist()
        target_pos = th.tensor([ 0.7194, -3.7099,  0.7962])
        
        # simulator.set_target_visual_pose([*target_pos,0,0,0],size=0.02)
        
        # 确定抓取的方向为 相机位置到 目标位置的方向
        camera_pos = simulator.camera.get_position_orientation()[0] # 0为位置,1为四元数
        grasp_direction = target_pos - camera_pos
        grasp_direction = grasp_direction / np.linalg.norm(grasp_direction)
        # target_euler 为 grasp_direction 的欧拉角  
        # target_euler = direction_vector_to_euler_angles(grasp_direction)
        # 计算欧拉角
        pitch = np.arctan2(grasp_direction[2], np.sqrt(grasp_direction[0]**2 + grasp_direction[1]**2))
        yaw = np.arctan2(grasp_direction[1], grasp_direction[0])
        roll = 0  # 假设 roll 为 0，因为我们只关心方向

        target_euler = [pitch, roll, yaw]
        
        # target_pos 为朝着 grasp_direction 方向的 1厘米 处
        target_pos = target_pos + grasp_direction * 0.02
        # 画出目标位置
        simulator.set_target_visual_pose([*target_pos,0,0,0],size=0.02)
        
        
        # target_quat = T.euler2quat(th.tensor(target_euler))
        # target_euler  state.target_euler
        state.target_local_pose = simulator.pose_to_local(target_pos.tolist()+target_euler)
        print(f"local pose: {state.target_local_pose}")
        success = simulator.grasp_object_by_pose(state.target_local_pose,object_name=cfg.target_object_name)
        if success:
            print(f"第{try_time}次尝试成功,保存数据")
            # add_hdf5_sample(cfg.hdf5_path,obs)
            success_rate += 1
            # data.append([i, point, True, True])
            break
        else:
            # data.append([i, point, False, False])
            pass
        print(f"第{try_time}次尝试失败")
    
    
    
    simulator.reset()
simulator.close()

print(f"{cfg.task_name} SR: {success_rate}/{total_try}")
print(f"{cfg.task_name} SR: {success_rate/total_try}*100%")

# save the data to csv
# 用 pd.DataFrame 保存
df = pd.DataFrame(data, columns=['try_num', 'point_correct', 'curobo_success', 'success'])
df.to_csv(f'{DIR}/../results/exp2_{cfg.task_name}_grasp_{total_try}.csv', index=False)
    
    

