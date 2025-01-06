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
from btgym.dataclass.cfg import cfg
# gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = False

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

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

        # self.load_empty_scene()
        # self.get_task_list()


    def load_empty_scene(self):
        config = {
            "env": {
            },
            "scene": {
                "type": "Scene",
                "trav_map_with_objects": False,  # 不生成导航地图
            },
            "robots": [],
            "objects": [],
            "task": {
                "type": "DummyTask"
            }
        }
        self.og_sim = og.Environment(configs=config, in_vec_env=False)


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

        config_filename = os.path.join(cfg.ASSETS_PATH, "fetch_primitives.yaml")
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



    def create_my_task(self, task_name):
        import omnigibson as og
        from bddl import config
        config.ACTIVITY_CONFIGS_PATH = f'{cfg.ASSETS_PATH}/my_tasks'
        from omnigibson.utils import bddl_utils 
        bddl_utils.BEHAVIOR_ACTIVITIES.append(task_name)

        config_filename = os.path.join(cfg.ASSETS_PATH, "fetch_primitives.yaml")
        # config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
        cfgs = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
        cfgs["scene"]["load_task_relevant_only"] = True

        cfgs["task"] = {
            "type": "BehaviorTask",
            "activity_name": task_name,
            "activity_definition_id": 0,
            "activity_instance_id": 0,
            "online_object_sampling": True,
        }
        env = og.Environment(configs=cfgs)

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

    def save_camera_image(self, output_path):
        """
        保存机器人视角的RGB图像
        Args:
            output_path: 输出图像的路径，例如 "robot_view.png"
        """
        rgb_obs = list(self.robot.get_obs()[0].values())[0]['rgb'].cpu().numpy()
        # 将numpy数组转换为PIL图像并保存
        from PIL import Image
        img = Image.fromarray(rgb_obs)
        img = img.convert('RGB')  # 将RGBA转换为RGB
        img.save(output_path, format='PNG')
    

if __name__ == "__main__":
    # print(gm.REMOTE_STREAMING)
    simulator = Simulator()
    simulator.load_behavior_task_by_name('putting_shoes_on_rack')
    # simulator.create_my_task('test_task')

    

    simulator.idle()

    # 调用大模型来执行任务
    # simulator.do_task()