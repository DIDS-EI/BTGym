
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

def main():
    """
    Demonstrates how to use the action primitives to pick and place an object in an empty scene.

    It loads Rs_int with a robot, and the robot picks and places a bottle of cologne.
    """
    
    # Load the config
    config_filename = os.path.join(os.path.dirname(__file__), "franka_primitives.yaml")
    # config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Update it to create a custom environment and run some actions
    config["scene"]["scene_model"] = "Rs_int"
    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls"]
    config["objects"] = [
        {
            "type": "DatasetObject",
            "name": "cologne",
            "category": "bottle_of_cologne",
            "model": "lyipur",
            "position": [0.5, -0.5, 0.6],
            "orientation": [0, 0, 0, 1],
        },
        {
            "type": "DatasetObject",
            "name": "table2",
            "category": "breakfast_table",
            "model": "rjgmmy",
            "scale": [0.3, 0.3, 0.7],
            "position": [0.5, -0.5, 0.2],
            "orientation": [0, 0, 0, 1],
        },
        {
            "type": "DatasetObject",
            "name": "table",
            "category": "breakfast_table",
            "model": "rjgmmy",
            "scale": [0.3, 0.3, 0.3],
            "position": [-0.3, 0.2, 0.2],
            "orientation": [0, 0, 0, 1],
        },
        
    ]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()
    
    # 获取抓取姿态 
    grasp_pose, object_direction = get_grasp_poses_for_object_sticky(scene.object_registry("name", "cologne"))[0]
    grasp_pos,grasp_quat = grasp_pose
    grasp_pos = grasp_pos + object_direction * 0.3
    # # 获取机器人当前末端执行器的位置和方向
    # current_pos = robot.get_eef_position().unsqueeze(0)  # 当前位置
    # current_quat = robot.get_eef_orientation().unsqueeze(0)  # 当前方向
    
    # 目标位置和方向
    target_pos = grasp_pos.unsqueeze(0)
    target_quat = grasp_quat.unsqueeze(0)
    
    # 将当前位置和目标位置拼接在一起
    pos_sequence = th.cat([target_pos, target_pos], dim=0)  # 形状变为 [2, 3]
    quat_sequence = th.cat([target_quat, target_quat], dim=0)  # 形状变为 [2, 4]
    


    # 创建运动规划器
    curobo_mg = CuRoboMotionGenerator(robot)
    # 计算运动轨迹
    #  使用运动规划器计算从当前位置到目标抓取姿态的轨迹，输入包括目标姿态 pose 和抓取方向 direction，返回的 trajectories 包含机器人关节的运动序列
    successes, paths = curobo_mg.compute_trajectories(pos_sequence, quat_sequence)
    print("paths:",paths)
    

    if successes[0]:  # 检查第一条轨迹是否成功
        # 将 JointState 转换为 joint trajectory tensor
        joint_trajectory = curobo_mg.path_to_joint_trajectory(paths[0])
        
        # 执行轨迹
        for time_i,joint_positions in enumerate(joint_trajectory):
            print(f"time_i: {time_i}, joint_positions: {joint_positions}")
            # 现在 joint_positions 已经是 tensor 格式
            # robot.set_joint_positions(joint_positions)
            # 更新模拟器
            env.step(joint_positions.to('cpu'))
            # og.sim.step()
            # time.sleep(0.1)
            
        print("轨迹执行完成!")
    else:
        print("轨迹规划失败!")

    
    # 继续运行模拟器（可选）
    while True:
        og.sim.step()
 

    # controller = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

    # # Grasp of cologne
    # grasp_obj = scene.object_registry("name", "cologne")
    # print("Executing controller")

    # primitive_action = controller.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, grasp_obj)

    # execute_controller(primitive_action, env)
    # print("Finished executing grasp")

    # # Place cologne on another table
    # print("Executing controller")
    # table = scene.object_registry("name", "table")
    # execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.PLACE_ON_TOP, table), env)
    # print("Finished executing place")
    print("Done!")

if __name__ == "__main__":
    main()
