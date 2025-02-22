import os

import yaml

import omnigibson as og

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
    # config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls"]

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

    while True: 
        og.sim.step()


if __name__ == "__main__":
    main()