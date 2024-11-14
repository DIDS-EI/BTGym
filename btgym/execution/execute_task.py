import json
import logging
import os
import time
import torch as th
import yaml

import omnigibson as og
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
from omnigibson.macros import gm
from btgym.utils.path import ROOT_PATH
from omnigibson.tasks.behavior_task import BehaviorTask

log_path = os.path.dirname(__file__)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建文件处理程序，将日志写入文件
file_handler = logging.FileHandler(f'{log_path}/logfile.log')
file_handler.setLevel(logging.INFO)

# 创建日志格式器并将其添加到处理程序
formatter = logging.Formatter('%(message)s')
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 将文件处理程序添加到日志记录器
logger.addHandler(file_handler)

def log(text):
    logger.info(text)

log('====== a new run ======')
log('time: ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))


action_map = {
    'grasp': StarterSemanticActionPrimitiveSet.GRASP,
    'place_on_top': StarterSemanticActionPrimitiveSet.PLACE_ON_TOP,
    'place_inside': StarterSemanticActionPrimitiveSet.PLACE_INSIDE,
    'open': StarterSemanticActionPrimitiveSet.OPEN,
    'close': StarterSemanticActionPrimitiveSet.CLOSE,
    'navigate_to': StarterSemanticActionPrimitiveSet.NAVIGATE_TO,
    'release': StarterSemanticActionPrimitiveSet.RELEASE,
    'toggle_on': StarterSemanticActionPrimitiveSet.TOGGLE_ON,
    'toggle_off': StarterSemanticActionPrimitiveSet.TOGGLE_OFF,
}

task_scene_map = json.load(open(f'{ROOT_PATH}/assets/task_to_scenes.json', 'r'))
print(len(task_scene_map))

# Don't use GPU dynamics and use flatcache for performance boost
# gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = False


def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)


def load_plan(plan_file):
    log('\nloading plan:')

    action_list = []
    with open(plan_file, 'r') as file:
        plan_lines = file.readlines()[:-1]
    for line in plan_lines:
        line_list = line.strip()[1:-1].split(' ')
        action_list.append(line_list)
        log(line_list)
    return action_list
    

def execute_task_single(task_name, plan_file):
    action_list = load_plan(plan_file)

    """
    Demonstrates how to use the action primitives to solve a simple BEHAVIOR-1K task.

    It loads Benevolence_1_int with a robot, and the robot attempts to solve the
    picking_up_trash task using a hardcoded sequence of primitives.
    """
    # Load the config
    config_filename = os.path.join(og.example_config_path, "fetch_primitives.yaml")
    # config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Update it to run a grocery shopping task
    config["scene"]["scene_model"] = task_scene_map[task_name][0]
    # config["scene"]["load_task_relevant_only"] = True
    # config["scene"]["not_load_object_categories"] = ["ceilings"]
    config["task"] = {
        "type": "BehaviorTask",
        "activity_name": task_name,
        "activity_definition_id": 0,
        "activity_instance_id": 0,
        "predefined_problem": None,
        "online_object_sampling": False,
    }

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    controller = StarterSemanticActionPrimitives(env, enable_head_tracking=False)


    log('\nexecuting plan:')
    for action in action_list:
        action_name = action[0]
        action_obj = action[-1]
        ground_obj = env.task.object_scope[action_obj]
        execute_controller(controller.apply_ref(action_map[action_name], ground_obj), env)
        log('finished executing action: ' + action)
    # Grasp can of soda
    # grasp_obj = env.task.object_scope["can__of__soda.n.01_2"]
    # print("Executing controller")
    # execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, grasp_obj), env)
    # print("Finished executing grasp")

    # Place can in trash can
    # print("Executing controller")
    # trash = env.task.object_scope["ashcan.n.01_1"]
    # execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.PLACE_INSIDE, trash), env)
    # print("Finished executing place")


if __name__ == "__main__":
    # execute_task_single('picking_up_trash',f'{ROOT_PATH}/../outputs/bddl_planning/success/picking_up_trash')
    start_idx = 8
    task_num = 1
    plan_folder = f'{ROOT_PATH}/../outputs/bddl_planning/success'
    for plan_file in os.listdir(plan_folder)[start_idx:start_idx+task_num]:
        log("执行task: " + plan_file)
        execute_task_single(plan_file, f'{plan_folder}/{plan_file}')
