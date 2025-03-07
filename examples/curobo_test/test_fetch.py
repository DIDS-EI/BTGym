
import numpy as np
from omnigibson.action_primitives.curobo import CuRoboMotionGenerator
import os

import yaml

import omnigibson as og

from omnigibson.macros import gm
import omnigibson.utils.transform_utils as T
import torch as th
import math
import os
import random
import time

DIRNAME = os.path.dirname(os.path.abspath(__file__))

# Don't use GPU dynamics and use flatcache for performance boost
# gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = True


def get_grasp_poses_for_object_sticky(target_obj):
    """
    Obtain a grasp pose for an object from top down, to be used with sticky grasping.

    Args:
        target_object (StatefulObject): Object to get a grasp pose for

    Returns:
        List of grasp candidates, where each grasp candidate is a tuple containing the grasp pose and the approach direction.
    """
    # 获取物体的边界框信息
    bbox_center_in_world, bbox_quat_in_world, bbox_extent_in_base_frame, _ = target_obj.get_base_aligned_bbox(
        visual=False
    )
    # 计算抓取中心位置
    grasp_center_pos = bbox_center_in_world + th.tensor([0, 0, th.max(bbox_extent_in_base_frame) + 0.05])
    # 计算抓取方向
    towards_object_in_world_frame = bbox_center_in_world - grasp_center_pos
    towards_object_in_world_frame /= th.norm(towards_object_in_world_frame)

    # 设置抓取方向（机器人末端执行器的姿态）
    # 修改前：侧面抓取
    grasp_quat = T.euler2quat(th.tensor([0, math.pi / 2, 0], dtype=th.float32))
    # 修改后：从上往下抓取
    # 另一种从上往下的姿态
    # grasp_quat = T.euler2quat(th.tensor([0, math.pi, 0], dtype=th.float32))


    grasp_pose = (grasp_center_pos, grasp_quat)
    grasp_candidate = [(grasp_pose, towards_object_in_world_frame)]

    return grasp_candidate

def main():
    # 加载配置文件
    config_filename = os.path.join(os.path.dirname(__file__), "assets/fetch_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # 设置场景
    config["scene"]["scene_model"] = "Rs_int"
    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls"]
    
    # 设置物体位置 - 确保在机器人工作空间内
    config["objects"] = [
        {
            "type": "DatasetObject",
            "name": "apple",
            "category": "apple",
            "model": "agveuv",
            "position": [0.6, 0, 0.5],  # 在机器人前方
            "orientation": [0, 0, 0, 1],
            "scale": [1, 1, 1],   # 小盒子更容易抓取
        },
        {
            "type": "DatasetObject",
            "name": "table",
            "category": "breakfast_table",
            "model": "rjgmmy",
            "scale": [0.3, 0.3, 0.5],
            "position": [0.6, 0, 0.2],   # 桌子在物体正下方
            "orientation": [0, 0, 0, 1],
        },
    ]

    # 加载环境
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]
    og.sim.enable_viewer_camera_teleoperation()

    # 等待环境稳定
    for _ in range(10):
        env.step(robot.get_joint_positions())
    
    # 获取目标物体
    target_obj = scene.object_registry("name", "apple")
    
    # 
    current_pos = robot.get_eef_position().unsqueeze(0)
    current_quat = robot.get_eef_orientation().unsqueeze(0)
    
    # 设置目标位置（在物体上方）
    target_pos = th.tensor([[0.5, 0, 1]], device=current_pos.device)  # 确保使用相同的device
    # 设置目标朝向（从上方抓取）
    # 从上往下
    target_quat = T.euler2quat(th.tensor([0, math.pi/2 , 0], device=current_pos.device)).unsqueeze(0)
    #  -math.pi/2
    
    # 打印位置信息
    print("Current position:", current_pos)
    print("Target position:", target_pos)
    print("Distance:", th.norm(target_pos - current_pos))
    
    # 构建轨迹序列
    pos_sequence = th.cat([target_pos, target_pos], dim=0)
    quat_sequence = th.cat([target_quat, target_quat], dim=0)
    
    # 创建运动规划器
    curobo_mg = CuRoboMotionGenerator(
        robot,
        robot_cfg_path=f"{DIRNAME}/assets/fetch_description_curobo.yaml"
    )
    
    # 计算轨迹
    print("Computing trajectory...")
    successes, paths = curobo_mg.compute_trajectories(pos_sequence, quat_sequence)
    print("Planning success:", successes)

    if successes[0]:
        
        # 在循环外获取一次初始关节位置
        # initial_joints = robot.get_joint_positions()
        # fixed_other_joints = initial_joints[:].clone()  # 克隆一份固定的非手臂关节位置 
        # print("固定的非手臂关节位置:", fixed_other_joints)
        
        
        # 执行轨迹
        joint_trajectory = curobo_mg.path_to_joint_trajectory(paths[0])
        for time_i,joint_positions in enumerate(joint_trajectory):
            # print(f"time_i: {time_i}, joint_positions: {joint_positions}")
            
            
            # 扩展关节位置到完整的14维
            # 创建正确的动作向量
            # 创建动作向量
            full_action = th.zeros(robot.n_joints, device=joint_positions.device)
            # 更新手臂关节（0-6）
            # full_action[:4] = fixed_other_joints[:4]
            # 使用固定的值设置其他关节
            full_action[5:12] = joint_positions[:7]
            # full_action[12:] = fixed_other_joints[12:]
            
            # Fetch机器人的关节顺序：
            # 0-3: 底盘
            # 4-11: 手臂
            # 12-13: 夹持器

            print(f"time_i: {time_i}, full_action: {full_action}")
            env.step(full_action.to('cpu'))
            time.sleep(0.01)  # 减慢执行速度以便观察
    
    # 保持场景运行
    while True:
        og.sim.step()
    #     env.step(robot.get_joint_positions())

if __name__ == "__main__":
    main()