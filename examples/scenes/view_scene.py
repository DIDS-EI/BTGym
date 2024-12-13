import btgym
import os
import queue
import multiprocessing

from btgym.utils.logger import set_logger_entry
import numpy as np
import yaml

import omnigibson as og
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
from omnigibson.macros import gm
from btgym.utils.logger import log
from omnigibson.robots.tiago import Tiago
from btgym.utils.path import ROOT_PATH
import json
import torch as th
import omnigibson.utils.transform_utils as T

gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = False

# gm.USE_GPU_DYNAMICS = False
# gm.ENABLE_FLATCACHE = True

def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)

task_scene_map = json.load(open(f'{ROOT_PATH}/assets/task_to_scenes.json', 'r'))

class Simulator:
    """
    Demonstrates how to use the action primitives to pick and place an object in an empty scene.
    """

    def __init__(self):
        self.og_sim = None
        self.current_task_name = None
        # Load the config
        # self.load_behavior_task_by_name('putting_shoes_on_rack')

        
        # self.null_control = np.zeros(self.robot.action_space.shape)
        self.control_queue = queue.Queue()
        # # Allow user to move camera more easily
        self.idle_control = np.array([ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.1700,  0.8585, -0.1485,
         1.8101,  1.6337,  0.1376, -1.3249, -0.6841,  0.0450,  0.0450,  0.8585,
        -0.1485,  1.8101,  1.6337,  0.1376, -1.3249, -0.6841,  0.0450,  0.0450])

        # self.get_task_list()

    def init_action_primitives(self):
        self.action_primitives = StarterSemanticActionPrimitives(self.og_sim, enable_head_tracking=False)

    def get_task_list(self):
        plan_folder = f'{ROOT_PATH}/../outputs/bddl_planning/success'
        self.task_list = os.listdir(plan_folder)
        # for plan_file in os.listdir(plan_folder):
        #     log("执行task: " + plan_file)
        #     execute_task_single(plan_file, f'{plan_folder}/{plan_file}')

        # task_name = 'putting_shoes_on_rack'
        # plan_file = f'{plan_folder}/{task_name}'
        # execute_task_single(task_name, plan_file)

    def load_behavior_task_by_index(self, task_index):
        task_name = self.task_list[task_index]
        self.load_behavior_task_by_name(task_name)

    def load_behavior_task_by_name(self, task_name):
        self.current_task_name = task_name
        log(f"load_behavior_task: {task_name}")

        config_filename = os.path.join(og.example_config_path, "fetch_primitives.yaml")
        # config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
        config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

        # Update it to run a grocery shopping task
        scene_name = task_scene_map[task_name][0]
        config["scene"]["scene_model"] = scene_name
        log(f'scene: {scene_name}')
        config["scene"]["load_task_relevant_only"] = True
        # config["scene"]["not_load_object_categories"] = ["ceilings"]
        config["task"] = {
            "type": "BehaviorTask",
            "activity_name": task_name,
            "activity_definition_id": 0,
            "activity_instance_id": 0,
            "predefined_problem": None,
            "online_object_sampling": False,
        }
        # config["robot"]["grasping_mode"] = "sticky"
        # gm.USE_GPU_DYNAMICS = True
        # gm.ENABLE_FLATCACHE = False


        # Load the environment
        if self.og_sim is None: 
            self.og_sim = og.Environment(configs=config)
        else:
            og.clear()
            og.sim.stop()
            self.og_sim.reload(configs=config)
            og.sim.play()
            self.og_sim.post_play_load()

        # self.reset()

        self.scene = self.og_sim.scene
        self.robot = self.og_sim.robots[0]

        # Allow user to move camera more easily
        # og.sim.enable_viewer_camera_teleoperation()

        # self.action_primitives = StarterSemanticActionPrimitives(self.og_sim, enable_head_tracking=False)
        og.sim.enable_viewer_camera_teleoperation()
        self.set_camera_lookat_robot()

    def reset(self):
        self.og_sim.reset()

    def step(self):
        if self.control_queue.empty():
            pass
            # self.og_sim.step(self.action_primitives._empty_action())
            # log('robot idle !!!')
        else:
            action = self.control_queue.get()
            if action is not None:
                self.og_sim.step(action)
            else:
                self.og_sim.step(self.action_primitives._empty_action())
                # log('robot step !!!')

    def set_viewer_camera_pose(self, position, orientation):
        og.sim.viewer_camera.set_position_orientation(position=position, orientation=orientation)

    def set_camera_lookat_robot(self):
        #设置
        robot_pos, robot_quat = self.robot.get_position_orientation()
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

    def idle(self):
        while True:
            og.sim.step()
            
    def add_control(self,control):
        self.control_queue.put(control)

    def get_scene_name(self):
        return self.scene.scene_model
    
    def get_robot_pos(self):
        return self.robot.get_position_orientation()[0]

    def get_trav_map(self):
        return self.scene._trav_map

    def do_task(self):
        log("start do_task")
        controller = StarterSemanticActionPrimitives(self.og_sim, enable_head_tracking=False)

        # Grasp of cologne
        grasp_obj = self.scene.object_registry("name", "cologne")
        print("Executing controller")

        primitive_action = controller.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, grasp_obj,attempts=10)


        self.add_control(primitive_action)
        # execute_controller(primitive_action, self.og_sim)
        # print("Finished executing grasp")

        # Place cologne on another table
        print("Executing controller")
        table = self.scene.object_registry("name", "table")
        primitive_action = controller.apply_ref(StarterSemanticActionPrimitiveSet.PLACE_ON_TOP, table,attempts=10)
        self.add_control(primitive_action)
        # execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.PLACE_ON_TOP, table), self.og_sim)
        # print("Finished executing place")

    def navigate_to_object(self, object_name):
        object = self.scene.object_registry("name", object_name)
        primitive_action = self.action_primitives.apply_ref(StarterSemanticActionPrimitiveSet.NAVIGATE_TO, object)
        self.add_control(primitive_action)

if __name__ == "__main__":
    # print(gm.REMOTE_STREAMING)
    simulator = Simulator()
    # simulator.load_behavior_task_by_name('putting_shoes_on_rack')
    # simulator.init_action_primitives()
    # gm.USE_GPU_DYNAMICS = True
    # gm.ENABLE_FLATCACHE = False
    #adding_chemicals_to_hot_tub
    simulator.load_behavior_task_by_name('folding_clothes')

    # scene_name = simulator.get_scene_name()
    # print(f"当前场景名称: {scene_name}")

    # robot_pos = simulator.get_robot_pos()
    # print(f"机器人位置: {robot_pos}")

    simulator.idle()
    # simulator.do_task()


# error
# tidy_your_garden

# correct
# buy_a_keg
# lighting_fireplace
# bringing_in_mail
# setting_up_room_for_games 把东西放抽屉里