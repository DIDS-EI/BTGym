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

from btgym.core.action_starter import action_starter_map as action_map
from btgym.core.action_starter import ActionPrimitives


from omnigibson.macros import gm
from btgym.utils.logger import log
from omnigibson.robots.tiago import Tiago
from btgym.utils.path import ROOT_PATH
import json
import torch as th
import omnigibson.utils.transform_utils as T

from btgym.core.curobo import CuRoboMotionGenerator
import math
from btgym.dataclass.cfg import cfg

import os
import btgym.utils.og_utils as og_utils
from bddl.object_taxonomy import ObjectTaxonomy
import time


OBJECT_TAXONOMY = ObjectTaxonomy()

task_list_path = os.path.join(cfg.ASSETS_PATH, 'tasks.txt')
VALID_TASK_LIST = open(task_list_path, 'r').read().splitlines()
VALID_SCENE_LIST = open(os.path.join(cfg.ASSETS_PATH, 'scene_list.txt'), 'r').read().splitlines()
# gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = False

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        if action:
            env.step(action)
        else:
            og.sim.step()

task_scene_map = json.load(open(f'{ROOT_PATH}/assets/task_to_scenes.json', 'r'))

class Simulator:
    """
    Demonstrates how to use the action primitives to pick and place an object in an empty scene.
    """

    def __init__(self,task_name='putting_shoes_on_rack',load_mode=None):
        self.og_sim = None
        self.current_task_name = None
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        
        # self.null_control = np.zeros(self.robot.action_space.shape)
        self.control_queue = queue.Queue()
        # # Allow user to move camera more easily
        self.idle_control = np.array([ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.1700,  0.8585, -0.1485,
         1.8101,  1.6337,  0.1376, -1.3249, -0.6841,  0.0450,  0.0450,  0.8585,
        -0.1485,  1.8101,  1.6337,  0.1376, -1.3249, -0.6841,  0.0450,  0.0450])
        self.action_primitives = None

        self.load_task(task_name,load_mode)

        from omni.isaac.core.objects import cuboid

        # Make a target to follow
        self.target_visual = cuboid.VisualCuboid(
            "/World/visual",
            position=np.array([0.3, 0, 0.67]),
            orientation=np.array([0, 1, 0, 0]),
            color=np.array([1.0, 0, 0]),
            size=0.2,
        )

    def idle(self):
        while True:
            og.sim.step()
    
    def idle_step(self):
        if og.sim:
            og.sim.step()



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
        self.og_sim = og.Environment(configs=config)



    def load_task(self, task_name=None):
        if task_name in VALID_TASK_LIST:
            self.load_behavior_task(task_name)
        else:
            self.load_empty_scene()

    def load_custom_task(self, task_name, scene_name=None, json_path=None,is_sample=False):
        import omnigibson as og
        from bddl import config
        config.ACTIVITY_CONFIGS_PATH = f'{cfg.ASSETS_PATH}/my_tasks'
        from omnigibson.utils import bddl_utils 
        bddl_utils.BEHAVIOR_ACTIVITIES.append(task_name)

        config_filename = os.path.join(cfg.ASSETS_PATH, "fetch_primitives.yaml")
        cfgs = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

        if is_sample:
            cfgs["scene"]["scene_model"] = scene_name
        else:   
            cfgs["scene"]["scene_file"] = json_path
        
        cfgs['task'] = {
                "type": "BehaviorTask",
                "activity_name": task_name,
                "activity_definition_id": 0,
                "activity_instance_id": 0,
                "online_object_sampling": is_sample,
            }
        self.load_from_config(cfgs)

    def sample_custom_task(self,task_name, scene_name=None):
        try:
            self.load_custom_task(task_name, scene_name=scene_name, is_sample=True)
        except Exception as e:
            print(f"Error loading task {task_name}: {str(e)}")
            return ''
        os.makedirs(f'{cfg.OUTPUTS_PATH}/sampled_tasks',exist_ok=True)
        json_path = f'{cfg.OUTPUTS_PATH}/sampled_tasks/{task_name}_{int(time.time())}.json'
        og.sim.save(json_paths=[json_path])
        return json_path


    def load_scene(self, scene_name):
        config_filename = os.path.join(cfg.ASSETS_PATH, "fetch_primitives.yaml")
        config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
        config["scene"]["scene_model"] = scene_name
        # config["scene"]["load_task_relevant_only"] = True
        # config["scene"]["not_load_object_categories"] = ["ceilings"]
        self.load_from_config(config)

    def load_from_config(self, config):
        # Load the environment
        if self.og_sim is None: 
            self.og_sim = og.Environment(configs=config)
        else:
            og.clear()
            og.sim.stop()
            self.og_sim.reload(configs=config)
            og.sim.play()
            self.og_sim.post_play_load()
    
        self.scene = self.og_sim.scene
        self.robot = self.og_sim.robots[0]

        # Allow user to move camera more easily
        # og.sim.enable_viewer_camera_teleoperation()

        if self.action_primitives is not None:
            del self.action_primitives

        self.action_primitives = ActionPrimitives(self)
        og.sim.enable_viewer_camera_teleoperation()
        self.reset_hand()
        self.set_camera_lookat_robot()

        self.curobo_mg = CuRoboMotionGenerator(self.robot,
            robot_cfg_path=f"{ROOT_PATH}/assets/fetch_description_curobo.yaml",
            debug=False)

        log("load task: success!")

        self.camera = list(self.robot.sensors.values())[0]

    # def load_from_json(self, task_name, json_path):
    #     config_filename = os.path.join(ROOT_PATH, "assets/fetch_primitives.yaml")
    #     config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    #     config["scene"]["scene_file"] = json_path

    #     self.load_from_config(config)

    # def load_from_json_task(self, json_path, task_name):
    #     pass
        # config_filename = os.path.join(cfg.ASSETS_PATH, "fetch_primitives.yaml")
        # config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
        # config["task"] = {
        #     "type": "BehaviorTask",
        #     "activity_name": task_name,
        #     "activity_definition_id": 0,
        #     "activity_instance_id": 0,
        #     "online_object_sampling": True,
        # }


    def load_behavior_task(self, task_name):
        self.current_task_name = task_name
        log(f"load_behavior_task: {task_name}")


        config_filename = os.path.join(cfg.ASSETS_PATH, "fetch_primitives.yaml")
        # config_filename = os.path.join(og.example_config_path, "fetch_primitives.yaml")
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

        self.load_from_config(config)

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


    def get_scene_name(self):
        return self.scene.scene_model
    
    def get_robot_pos(self):
        return self.robot.get_position_orientation()[0]

    def get_trav_map(self):
        return self.scene._trav_map

    def get_joint_states(self):
        return self.robot.get_joint_positions()

    def set_joint_states(self,joint_states):
        joint_states = th.tensor(joint_states,device=self.device)
        self.robot.set_joint_positions(joint_states)

    def get_end_effector_pose(self):
        return self.robot.get_eef_position_orientation()

    def get_relative_eef_pose(self):
        return self.robot.get_relative_eef_pose()

    def get_task_objects(self):
        return list(self.og_sim.task.object_instance_to_category.keys())

    def navigate_to_object(self, object_name):
        # object = self.scene.object_registry("name", object_name)
        object = self.og_sim.task.object_scope[object_name]
        primitive_action = self.action_primitives.apply_ref(StarterSemanticActionPrimitiveSet.NAVIGATE_TO, object)
        execute_controller(primitive_action, self.og_sim)

    def grasp_object(self, object_name):
        object = self.og_sim.task.object_scope[object_name]
        primitive_action = self.action_primitives.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, object)
        execute_controller(primitive_action, self.og_sim)


    def add_control(self,control):
        self.control_queue.put(control)


    def reach_pose(self, pose, is_local=False):
        robot = self.robot
        pos = th.tensor(pose[:3],device=self.device)
        euler = th.tensor(pose[3:],device=self.device)

        if euler is None:
            quat = T.euler2quat(th.tensor([0,math.pi/2,0], dtype=th.float32)) # 默认��上往下抓
        else:
            quat = T.euler2quat(euler)
        
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

        successes, paths = self.curobo_mg.compute_trajectories(pos_sequence, quat_sequence,is_local=is_local)

        if successes[0]:
            # 执行轨迹
            joint_trajectory = self.curobo_mg.path_to_joint_trajectory(paths[0])
            # 打印轨迹
            print(joint_trajectory)
            
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
                self.og_sim.step(full_action.to('cpu'))

    def reset_hand(self):
        jp = self.get_joint_states()
        self.set_joint_states([
                0.0,
                0.0,  # wheels
                0.0,  # trunk
                0.0,
                -1.5,
                0.0,  # head
                -0.8,
                1.7,
                2.0,
                -1.0,
                1.36904,
                1.90996,  # arm
                jp[-2],
                jp[-1],  # gripper
            ])
        
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
    
    def set_target_visual_pose(self, pose):
        pos = th.tensor(pose[:3],device=self.device)
        euler = th.tensor(pose[3:],device=self.device)
        if euler is None:
            quat = T.euler2quat(th.tensor([0,math.pi/2,0], dtype=th.float32))
        else:
            quat = T.euler2quat(euler)
        
        self.target_visual.set_world_pose(pos, quat)

    def set_camera_lookat_pos(self,pos):
        z_limits = [-1.57, 1.57]
        y_limits = [-0.76, 1.45]
        
        target_position = th.tensor(pos)
        current_camera_position = self.camera.get_position_orientation()[0]
        current_robot_quat = self.robot.get_position_orientation()[1]
        robot_euler = T.quat2euler(current_robot_quat)
        robot_euler[0] = 0.0
        robot_euler[1] = 0.0
        direction = target_position - current_camera_position
        
        # 考虑机器人朝向x轴正方向的情况
        cos_yaw = th.cos(robot_euler[2])
        sin_yaw = th.sin(robot_euler[2])
        
        # 修改：交换x和y的角色，因为机器人朝向x轴
        rel_x = direction[1] * cos_yaw - direction[0] * sin_yaw
        rel_y = direction[0] * cos_yaw + direction[1] * sin_yaw
        rel_z = direction[2]
        
        target_z_angle = th.atan2(rel_x, rel_y)
        distance_horizontal = th.sqrt(rel_x**2 + rel_y**2)
        target_y_angle = -th.atan2(rel_z, distance_horizontal)
        
        target_z_angle = th.clamp(target_z_angle, min=z_limits[0], max=z_limits[1])
        target_y_angle = th.clamp(target_y_angle, min=y_limits[0], max=y_limits[1])
        
        joint_state = self.get_joint_states()
        joint_state[3] = target_z_angle
        joint_state[5] = target_y_angle
        print(f"target_z_angle: {target_z_angle}, target_y_angle: {target_y_angle}")
        self.robot.set_joint_positions(joint_state)

        # self.set_camear_lookat_pos_ori(pos)


    #TODO 目前无法使物体在相机的正中心，有待修正
    def set_camear_lookat_pos_ori(self,pos):
        pos = [0,0,1]
        v1 = [0,0,-1]
        target_pos = th.tensor(pos)
        current_pos = self.camera.get_position_orientation()[0]
        v2 = target_pos - current_pos


        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # 计算旋转轴
        axis = np.cross(v1, v2)
        axis = axis / np.linalg.norm(axis)
        
        # 计算旋转角度
        angle = np.arccos(np.dot(v1, v2))
        
        # 计算四元数的各个分量
        w = math.cos(angle / 2.0)
        x = axis[0] * math.sin(angle / 2.0)
        y = axis[1] * math.sin(angle / 2.0)
        z = axis[2] * math.sin(angle / 2.0)
        # 将旋转矩阵转换为四元数
        quat = th.tensor([w,x,y,z])
        
        self.camera.set_position_orientation(current_pos, quat)



    def get_camera_info(self):
        sensor = list(self.robot.sensors.values())[0]
        intrinsics = sensor.intrinsic_matrix.reshape(-1).cpu().numpy()
        pose = sensor.get_position_orientation()
        extrinsics = T.pose_inv(T.pose2mat(pose)).reshape(-1).cpu().numpy()

        return {
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
        }


    def get_obs(self):
        sensor_obs = list(self.robot.sensors.values())[0].get_obs()
        rgb_obs = sensor_obs[0]['rgb'].cpu().numpy().tobytes()
        depth_obs = sensor_obs[0]['depth_linear'].cpu().numpy().tobytes()
        seg_obs = sensor_obs[0]['seg_semantic'].cpu().numpy().tobytes()
        seg_info = json.dumps(sensor_obs[1]['seg_semantic'])
        proprio_obs = self.robot.get_proprioception()[0].cpu().numpy()

        return {
            'rgb': rgb_obs,
            'depth': depth_obs,
            'seg_semantic': seg_obs,
            'seg_info': seg_info,
            'proprio': proprio_obs,
        }


    def get_available_objects(self):
        exclude_prefix = ['wall', 'floor', 'ceilings']
        fixed_object_names = set(self.scene.fixed_objects)

        exclude_objects = []
        fixed_objects = []
        moveable_objects = []
        for obj in self.scene.objects:
            category = obj.category
            if any(category.startswith(prefix) for prefix in exclude_prefix):
                exclude_objects.append(obj.name)
            elif obj.name in fixed_object_names:
                fixed_objects.append(obj.name)
                synset = OBJECT_TAXONOMY.get_synset_from_category(obj.category)
                print(f'fixed: {synset}')
            else:
                moveable_objects.append(obj.name)
                synset = OBJECT_TAXONOMY.get_synset_from_category(obj.category)
                print(f'moveable: {synset}')

        return {'moveable_objects': moveable_objects,
                'fixed_objects': fixed_objects}



if __name__ == "__main__":
    # print(gm.REMOTE_STREAMING)
    simulator = Simulator(task_name='test_task', load_mode='sample_task')
    # simulator = Simulator('Rs_int')
    # simulator = Simulator('putting_shoes_on_rack')
    # simulator.load_behavior_task_by_name('putting_shoes_on_rack')
    # simulator.init_action_primitives()
    # gm.USE_GPU_DYNAMICS = True
    # gm.ENABLE_FLATCACHE = False
    #adding_chemicals_to_hot_tub
    # simulator.load_behavior_task_by_name('folding_clothes')

    # scene_name = simulator.get_scene_name()
    # print(f"当前场景名称: {scene_name}")

    # robot_pos = simulator.get_robot_pos()
    # print(f"机器人位置: {robot_pos}")
    # simulator.get_obs()
    # camera_info = simulator.get_camera_info()
    # print(f"相机信息: {camera_info}")

    # available_objects = simulator.get_available_objects()
    # print(f"可用的物体: {available_objects}")
    simulator.idle()
    # simulator.do_task()


# error
# tidy_your_garden

# correct
# buy_a_keg
# lighting_fireplace
# bringing_in_mail
# setting_up_room_for_games 把东西放抽屉里