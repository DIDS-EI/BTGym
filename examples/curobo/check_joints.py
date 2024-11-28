
from btgym.core.curobo import CuRoboMotionGenerator
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

from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
import numpy as np

# Don't use GPU dynamics and use flatcache for performance boost
# gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = True

def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        # print(f"action: {action}")
        env.step(action)


def get_grasp_poses_for_object_sticky(target_obj):
    """
    Obtain a grasp pose for an object from top down, to be used with sticky grasping.

    Args:
        target_object (StatefulObject): Object to get a grasp pose for

    Returns:
        List of grasp candidates, where each grasp candidate is a tuple containing the grasp pose and the approach direction.
    """
    bbox_center_in_world, bbox_quat_in_world, bbox_extent_in_base_frame, _ = target_obj.get_base_aligned_bbox(
        visual=False
    )

    grasp_center_pos = bbox_center_in_world + th.tensor([0, 0, th.max(bbox_extent_in_base_frame) + 0.05])
    towards_object_in_world_frame = bbox_center_in_world - grasp_center_pos
    towards_object_in_world_frame /= th.norm(towards_object_in_world_frame)

    grasp_quat = T.euler2quat(th.tensor([0, math.pi / 2, 0], dtype=th.float32))

    grasp_pose = (grasp_center_pos, grasp_quat)
    grasp_candidate = [(grasp_pose, towards_object_in_world_frame)]

    return grasp_candidate


def open_grisper(env):
    robot = env.robots[0]
    current_joint_positions = robot.get_joint_positions()
    current_joint_positions[-1] = 1
    current_joint_positions[-2] = 1
    print(f"current_joint_positions: {current_joint_positions}")
    for i in range(100):
        env.step(current_joint_positions)
    # print(f"current_joint_positions: {current_joint_positions}")

def close_grisper(env):
    robot = env.robots[0]
    current_joint_positions = robot.get_joint_positions()
    current_joint_positions[-1] = -1
    current_joint_positions[-2] = -1
    print(f"current_joint_positions: {current_joint_positions}")
    for i in range(100):
        env.step(current_joint_positions)
    # print(f"current_joint_positions: {current_joint_positions}")



def reach_object(env, curobo_mg, obj_name, offest):
    scene = env.scene
    grasp_obj = scene.object_registry("name", obj_name)
    # 获取抓取姿态 
    grasp_pose, object_direction = get_grasp_poses_for_object_sticky(grasp_obj)[0]
    grasp_pos,grasp_quat = grasp_pose
    reach_pose(env, curobo_mg, grasp_pos, grasp_quat)

def reach_pose(env, curobo_mg, pos,quat=None):
    robot = env.robots[0]
    if quat is None:
        quat = T.euler2quat(th.tensor([0,math.pi,0], dtype=th.float32))

    # 将当前位置和目标位置拼接在一起
    pos_sequence = th.stack([pos, pos])  # 形状变为 [2, 3]
    quat_sequence = th.stack([quat, quat])  # 形状变为 [2, 4]

    # 如果机器人接近关节限制，则调整关节位置
    jp = robot.get_joint_positions(normalized=True)
    if not th.all(th.abs(jp)[:-2] < 0.97):
        new_jp = jp.clone()
        new_jp[:-2] = th.clamp(new_jp[:-2], min=-0.95, max=0.95)
        robot.set_joint_positions(new_jp, normalized=True)
    og.sim.step()

    successes, paths = curobo_mg.compute_trajectories(pos_sequence, quat_sequence)
    # # print("paths:",paths)


    if successes[0]:

        # 执行轨迹
        joint_trajectory = curobo_mg.path_to_joint_trajectory(paths[0])

        print(joint_trajectory)
        for time_i,joint_positions in enumerate(joint_trajectory):
            # joint_positions = joint_trajectory[-1]
            full_action = th.zeros(robot.n_joints, device=joint_positions.device)

            full_action[2] = joint_positions[0]
            full_action[5:] = joint_positions[1:]
            
            # robot.set_joint_positions(full_action)
            
            print(f"time_i: {time_i}, full_action: {full_action}")
            env.step(full_action.to('cpu'))


        # env.step(full_action.to('cpu'))

    # if successes[0]:  # 检查第一条轨迹是否成功
    #     # 将 JointState 转换为 joint trajectory tensor
    #     joint_trajectory = curobo_mg.path_to_joint_trajectory(paths[0])
        
    #     # 执行轨迹
    #     for time_i,joint_positions in enumerate(joint_trajectory):
    #         # print(f"time_i: {time_i}, joint_positions: {joint_positions}")
    #         # 现在 joint_positions 已经是 tensor 格式
    #         # robot.set_joint_positions(joint_positions)
    #         # 更新模拟器
    #         env.step(joint_positions.to('cpu'))
    #         # og.sim.step()
    #         # time.sleep(0.1)
            
    #     print("轨迹执行完成!")
    # else:
    #     print("轨迹规划失败!")

def reset_robot(env):
    robot = env.robots[0]
    robot.set_joint_positions(th.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), normalized=True)
    og.sim.step()

def main():
    """
    Demonstrates how to use the action primitives to pick and place an object in an empty scene.

    It loads Rs_int with a robot, and the robot picks and places a bottle of cologne.
    """
    
    # Load the config
    # config_filename = os.path.join(os.path.dirname(__file__), "assets/franka_primitives.yaml")
    config_filename = os.path.join(os.path.dirname(__file__), "assets/fetch_primitives.yaml")
    # config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Update it to create a custom environment and run some actions
    config["scene"]["scene_model"] = "Rs_int"
    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls"]


    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]



    print("start task!!!")
    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()
    
    action_space = robot.action_space
    action = np.zeros(action_space.shape[0])

    env.step(action)

    # # 如果上下限是inf,就设为100/-100
    # action_space.high[action_space.high == np.inf] = 100
    # action_space.low[action_space.low == -np.inf] = -100
    
    # # 每个维度生成12个等间隔值
    # samples_per_dim = 20
    # steps_per_action = 8
    
    # # 遍历每个维度
    # for dim in range(action_shape):
    #     print(f"dim: {dim}")
    #     # 生成当前维度的等间隔值
    #     values = np.linspace(action_space.low[dim], action_space.high[dim], samples_per_dim)[1:-1]
        
    #     # 对每个值执行动作
    #     for value in values:
    #         # 创建动作数组,将当前维度设为指定值,其他维度为0
    #         action = np.zeros(action_shape)
    #         action[dim] = value
            
    #         # 执行steps_per_action次
    #         for _ in range(steps_per_action):
    #             env.step(action)


    while True:
        og.sim.step()

    # print("Done!")

if __name__ == "__main__":
    main()
