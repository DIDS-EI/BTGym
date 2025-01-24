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
from omnigibson.macros import macros as m
from omnigibson import object_states
cfg.task_name='switch2'
cfg.scene_file_name='scene_file_0'
cfg.target_object_name = 'oven.n.01_1' #'electric_switch_wseglt_0' #'apple.n.01_1'
current_dir = Path(__file__).parent
cfg.hdf5_path = current_dir / 'robot_data.hdf5'

"""
# xxx. 保存成功数据
"""


if __name__ == '__main__':
    # sample_task()
    # restart_simulator()
    
    simulator = Simulator(headless=False)

    simulator.load_custom_task(task_name=cfg.task_name, scene_file_name=cfg.scene_file_name)

    for i in range(10):
        simulator.navigate_to_object(object_name=cfg.target_object_name)
        obs = simulator.get_obs()
        obs['gripper_open'] = False
        obs['eef_pose'] = state.target_local_pose
        
        # # # ========= 打个点 有点效果
        # rgb_img = Image.fromarray(obs['rgb'])
        # rgb_img.save(f'{CURRENT_DIR}/camera_grasp_rgb.png')
        # molmo_client = MolmoClient()
        # query = f'point out the red button of the {cfg.target_object_name.split(".")[0]}.'
        # point = molmo_client.get_grasp_pose_by_molmo(query,CURRENT_DIR,point_img_path=f'{CURRENT_DIR}/camera_grasp_rgb.png')

        
        # if not point:
        #     continue
        
        # # point 转为机器人相对坐标
        # camera_info = simulator.get_camera_info()
        # # point 转为机器人相对坐标
        # target_pos = og_utils.pixel_to_world(obs, camera_info, point[0], point[1])
        # print("target_pos:",target_pos)
        # # ========= 
        obj = simulator.og_sim.task.object_scope[cfg.target_object_name]
        toggle_state = obj.states[object_states.ToggledOn]
        # toggle_position = toggle_state.get_link_position()
        toggle_position = list(toggle_state._links.values())[0].aabb_center
        target_pos = toggle_position
        # target_pos = th.tensor([-1.6539, -3.0124,  0.9318])
        
        #  可视化目标点

        
        simulator.navigate_to_pos(object_name=cfg.target_object_name,pos=target_pos,offset=(0.9,-0.2))
        
        
        # 获取朝向
        # obj_face_tensor = simulator.get_object_face_tensor(object_name=cfg.target_object_name,pos=target_pos,horizontal=True)
        obj_face_tensor = th.tensor([0,1,0],dtype=th.float32)
        yaw = math.atan2(-obj_face_tensor[1],obj_face_tensor[0])
        
        # 再次导航到目标点
        # target_pos = th.tensor([1.5, -8.7688,  0.62])
        # simulator.navigate_to_pos(object_name=cfg.target_object_name,pos=target_pos,offset=(0.9,-0.2))
        target_pos = target_pos+obj_face_tensor*0.1
        target_euler = [math.pi/2, 0, yaw]
        

        
        # simulator.set_target_visual_pose([*toggle_position,0,0,0],size=0.1)
        simulator.idle_step(10)
        
        # 达到 point 的位置
        # target_local_pose = simulator.pose_to_local([*target_pos, *target_euler])
        # simulator.close_gripper()
        target_local_pose = simulator.pose_to_local([*target_pos, *target_euler])
        success = simulator.reach_pose(target_local_pose,is_local=True)
        
        # 往外面推？
        simulator.close_gripper()
        simulator.move_hand_linearly(dir=-obj_face_tensor,distance=0.5)
        
        steps = m.object_states.toggle.CAN_TOGGLE_STEPS
        for _ in range(steps+5):
            simulator.idle_step(1)
        # obj = simulator.get_object(cfg.target_object_name)
        # print("obj.states[ToggledOn].get_value():",obj.states[ToggledOn].get_value())
        simulator.idle_step(10)
        
        if success:
            # 用 BDDL 判断是否成功
            goal_list = simulator.og_sim.task.ground_goal_state_options[0]
            for goal in goal_list:
                print("goal.terms:",goal.terms)
                print("goal.currently_satisfied:",goal.currently_satisfied)
                
                if goal.terms[0] == 'toggled_on':
                    obj = simulator.og_sim.task.object_scope[goal.terms[1]]
                    toggle_state = obj.states[object_states.ToggledOn]
                    # toggle_position = toggle_state.get_link_position()
                    toggle_position = list(toggle_state._links.values())[0].aabb_center 
                    # eef_pos = simulator.get_end_effector_pose()[0]
                    eef_pos = simulator.robot.links['gripper_link'].aabb_center 
                    dis = np.linalg.norm(eef_pos-toggle_position)
                    if dis < 0.5:
                        print("dis:",dis)
                        print("成功")
                        break
            
            
            # add_hdf5_sample(cfg.hdf5_path,obs)
            break

        print(f"第{i}次尝试失败")



# 根据 bddl 判断是否完成
# simulator.og_sim.task.ground_goal_state_options[0][0].currently_satisfied 
goal_list = simulator.og_sim.task.ground_goal_state_options[0]
# print("goal_list[0].terms:",goal_list[0].terms) #['ontop', 'printer.n.03_1', 'table.n.02_1']
# print("goal_list[0].currently_satisfied:",goal_list[0].currently_satisfied)
for goal in goal_list:
    print("goal.terms:",goal.terms) #['ontop', 'printer.n.03_1', 'table.n.02_1']
    print("goal.currently_satisfied:",goal.currently_satisfied)


simulator.idle()
simulator.close()

    
    # 保存数据

    # 保存图像
    # 保存点
    # 保存成功与否
