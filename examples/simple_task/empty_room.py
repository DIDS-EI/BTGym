import os

import yaml

import omnigibson as og
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
from omnigibson.macros import gm
import torch as th
import time
import omnigibson.utils.transform_utils as T
# Don't use GPU dynamics and use flatcache for performance boost
# gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = True


def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)


def main():
    """
    Demonstrates how to use the action primitives to pick and place an object in an empty scene.

    It loads Rs_int with a robot, and the robot picks and places a bottle of cologne.
    """



    # Load the config
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Update it to create a custom environment and run some actions
    config["scene"]["scene_model"] = "Rs_int"
    # config["scene"]["load_object_categories"] = ["floors"]
    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls"]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

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


    last_time = time.time()
    while True:
        env.step(th.zeros(robot.action_dim))
        pos, quat = og.sim.viewer_camera.get_position_orientation()
        euler = T.quat2euler(quat).numpy()
        
        if time.time() - last_time > 3.0:
            print(f"{pos}", f"{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}")
            last_time = time.time()
        
            #设置
            robot_pos, robot_quat = robot.get_position_orientation()
            robot_euler = T.quat2euler(robot_quat)

            print(robot_euler)
            # 计算机器人前方向量
            forward_dir = T.quat2mat(robot_quat)[:3, 0]  # 取旋转矩阵的第一列作为前方向量
            # 计算前方1米的位置
            camera_pos = robot_pos + forward_dir * 1.0
            camera_pos[2] = 2

            # 计算与y轴的夹角
            y_axis = th.tensor([0.0, 1.0, 0.0])
            angle = th.acos(th.dot(-forward_dir, y_axis) / (th.norm(-forward_dir) * th.norm(y_axis)))
            angle = th.sign(forward_dir[0])*angle

            camera_quat = T.euler2quat(th.tensor([0.45,0,angle]))
            og.sim.viewer_camera.set_position_orientation(camera_pos, camera_quat)

if __name__ == "__main__":
    main()
