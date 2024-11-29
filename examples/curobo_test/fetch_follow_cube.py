
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
    # current_joint_positions = robot.get_joint_positions()
    current_joint_positions = th.zeros(robot.n_joints)
    current_joint_positions[-1] = 0.05 #1
    current_joint_positions[-2] = 0.05 #1
    print(f"current_joint_positions: {current_joint_positions}")
    for i in range(100):
        env.step(current_joint_positions)
    # print(f"current_joint_positions: {current_joint_positions}")

def close_grisper(env,joint_positions):
    robot = env.robots[0]
    current_joint_positions = th.zeros(robot.n_joints, device=joint_positions.device)
    current_joint_positions[4:] = joint_positions
    current_joint_positions[-1] = 0.0 #-1
    current_joint_positions[-2] = 0.0 #-1
    print(f"current_joint_positions: {current_joint_positions}")
    for i in range(100):
        env.step(current_joint_positions)
    # print(f"current_joint_positions: {current_joint_positions}")

# def is_object_grasped(env, target=None):
#     """检查物体是否被抓取"""
#     scene = env.scene
#     robot = env.robots[0]
#     # if obj_name!=None:
#     #     target_obj = scene.object_registry("name", obj_name)
#     cube_position, cube_orientation = target.get_world_pose()
    
#     # 获取夹爪的碰撞链接
#     l_finger = robot.links["l_gripper_finger_link"]
#     r_finger = robot.links["r_gripper_finger_link"]
    
#     # 检查物体是否与夹爪接触
#     contacts_l = l_finger.get_contacts()
#     contacts_r = r_finger.get_contacts()
    
#     # 检查接触的物体是否是目标物体
#     for contact in contacts_l + contacts_r:
#         if contact.other_link.prim_path == target_obj.prim_path:
#             return True
#     return False


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
        # math.pi/2 默认从上往下抓
        # math.pi 从左往右边抓
        quat = T.euler2quat(th.tensor([0,math.pi/2,0], dtype=th.float32)) # 默认从上往下抓
    

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

    # 1. 固定方向抓
    # successes, paths = curobo_mg.compute_trajectories(pos_sequence, quat_sequence)
    # # print("paths:",paths)
    
    # 2. 不固定方向抓
    # 定义多个可能的抓取朝向
    grasp_orientations = [
        T.euler2quat(th.tensor([0, math.pi/2, 0], dtype=th.float32)),      # 从上往下抓
        T.euler2quat(th.tensor([0, -math.pi/2, 0], dtype=th.float32)),     # 从下往上抓
        T.euler2quat(th.tensor([0, 0, 0], dtype=th.float32)),              # 从前往后抓
        T.euler2quat(th.tensor([0, math.pi, 0], dtype=th.float32)),        # 从后往前抓
        T.euler2quat(th.tensor([0, 0, math.pi/2], dtype=th.float32)),      # 从左往右抓
        T.euler2quat(th.tensor([0, 0, -math.pi/2], dtype=th.float32)),     # 从右往左抓
        # T.euler2quat(th.tensor([0, 3*math.pi/4, 0], dtype=th.float32)) # 45度角抓取
    ]
    for grasp_quat in grasp_orientations:
        pos_sequence = th.stack([pos, pos])
        quat_sequence = th.stack([grasp_quat, grasp_quat])
        try:
            successes, paths = curobo_mg.compute_trajectories(pos_sequence, quat_sequence)
            if successes[0]:
                break
        except Exception as e:  # 规划失败，换个抓取方向再规划
            print(f"Error: {e}")
            continue


    if successes[0]:

        # 执行轨迹
        joint_trajectory = curobo_mg.path_to_joint_trajectory(paths[0])
        # 打印轨迹
        print(joint_trajectory)
        
        # 打开夹爪
        # open_grisper(env)
        
        for time_i,joint_positions in enumerate(joint_trajectory):
            # joint_positions = joint_trajectory[-1]
            full_action = th.zeros(robot.n_joints, device=joint_positions.device)

            # full_action[2] = joint_positions[0]
            full_action[4:] = joint_positions
            
            # robot.set_joint_positions(full_action)
            
            if time_i == len(joint_trajectory) - 1:
                full_action[-1] = 0.0
                full_action[-2] = 0.0
            else:
                full_action[-1] = 0.05
                full_action[-2] = 0.05  

            print(f"time_i: {time_i}, full_action: {full_action}")
            env.step(full_action.to('cpu'))
        
        # 检查是否抓取成功
        # if is_object_grasped(env):
        #     print("抓取成功!")
            

        # # 关闭夹爪
        # close_grisper(env,joint_positions)


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
    


    # grasp_pos = grasp_pos + object_direction * 0.2
    # # 获取机器人当前末端执行器的位置和方向
    # current_pos = robot.get_eef_position().unsqueeze(0)  # 当前位置
    # current_quat = robot.get_eef_orientation().unsqueeze(0)  # 当前方向

    from omni.isaac.core.objects import cuboid

    # Make a target to follow
    target = cuboid.VisualCuboid(
        "/World/visual",
        position=np.array([0.3, 0, 0.67]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )

    DIRNAME = os.path.dirname(os.path.abspath(__file__))
    
    curobo_mg = CuRoboMotionGenerator(robot,
        robot_cfg_path=f"{DIRNAME}/assets/fetch_description_curobo.yaml",
        debug=False)

    # 修改
    # 修改夹爪位置
    curobo_mg.mg.kinematics.lock_joints = {
        "r_gripper_finger_joint": 0.0,  # 新的夹爪位置
        "l_gripper_finger_joint": 0.0   # 新的夹爪位置
    }


    # execute_controller(controller._execute_release(), env)

    n = 0
    while True:
        # for action in action_list:
            # execute_controller(action, env)
        cube_position, cube_orientation = target.get_world_pose()
        try:
            reach_pose(env, curobo_mg, th.tensor(cube_position), th.tensor(cube_orientation))
            # break
        except Exception as e:
            n+=1
            if n > 10:
                reset_robot(env)
                n = 0
            print(f"Error: {e}")

        # try:    
        #     grasp_pos = [0.7,0,1.418]
        #     print(f"grasp_pos: {grasp_pos}")
        #     reach_object(env, curobo_mg, "cologne", th.tensor(grasp_pos))
        #     break
        # except Exception as e:
        #     n += 0.002
        #     print(f"Error: {e}")
        og.sim.step()
    
    # print(grasp_pos)
    # print(grasp_pos)
    # close_grisper(env)
    # execute_controller(controller._execute_grasp(), env)
    # execute_controller(controller._empty_action(), env)

    while True:
        og.sim.step()

    # print("Done!")

if __name__ == "__main__":
    main()
