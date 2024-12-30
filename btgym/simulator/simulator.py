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

# from btgym.core.curobo import CuRoboMotionGenerator
from omnigibson.action_primitives.curobo import CuRoboMotionGenerator   
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
        if action!=None:
            env.step(action)
        else:
            og.sim.step()

task_scene_map = json.load(open(f'{ROOT_PATH}/assets/task_to_scenes.json', 'r'))

class Simulator:
    """
    Demonstrates how to use the action primitives to pick and place an object in an empty scene.
    """

    def __init__(self,task_name=None):
        self.og_sim = None
        self.current_task_name = None
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        
        self.robot = None
        self.scene = None
        self.action_primitives = None
        self.curobo_mg = None
        self.camera = None
        self.target_visual = None
        
        self.load_task(task_name)


    def idle(self):
        while True:
            og.sim.step()
    
    def idle_step(self,step_num=1):
        for _ in range(step_num):
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
        # self.action_primitive = StarterSemanticActionPrimitives(self.og_sim,enable_head_tracking=False)




    def load_task(self, task_name=None):
        if task_name in VALID_TASK_LIST:
            self.load_behavior_task(task_name)
        else:
            self.load_empty_scene()


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


    def load_custom_task(self, task_name, scene_name=None, scene_file_name=None,is_sample=False):
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
            cfgs["scene"]["scene_file"] = f'{cfg.task_folder}/{task_name}/{scene_file_name}.json'
        
        if not is_sample:
            cfgs["scene"]["load_task_relevant_only"] = True

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

        self.og_sim.task.write_task_metadata()
        # og.sim.write_metadata('scene_file_name',scene_name)
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
            og.sim.stop()
            og.clear()
            
            # 确保在重新加载前清理所有相机相关资源
            if self.action_primitives is not None:
                del self.action_primitives
            if self.curobo_mg is not None:
                del self.curobo_mg
            if self.target_visual is not None:
                del self.target_visual
            if self.camera is not None:
                del self.camera
            
            self.og_sim.reload(configs=config)
            og.sim.play()
            self.og_sim.post_play_load()

        # 等待几帧确保相机初始化完成
        for _ in range(10):
            self.idle_step()
        
        # 然后再设置相机
        og.sim.enable_viewer_camera_teleoperation()

        self.scene = self.og_sim.scene
        self.robot = self.og_sim.robots[0]
        self.camera = list(self.robot.sensors.values())[0]

        # Allow user to move camera more easily
        # og.sim.enable_viewer_camera_teleoperation()

        self.action_primitives = ActionPrimitives(self)
        self.reset_hand()
        self.set_camera_lookat_robot()

        self.curobo_mg = CuRoboMotionGenerator(self.robot,
            robot_cfg_path=f"{ROOT_PATH}/assets/fetch_description_curobo.yaml",
            debug=False)
        self.kinematics_config = self.curobo_mg.mg.robot_cfg.kinematics.kinematics_config

        log("load task: success!")

        from omni.isaac.core.objects import cuboid

        # Make a target to follow
        self.target_visual = cuboid.VisualCuboid(
            "/World/visual",
            position=np.array([0.3, 0, 0.67]),
            orientation=np.array([0, 1, 0, 0]),
            color=np.array([1.0, 0, 0]),
            size=0.01,
        )

        for i in range(10):
            self.idle_step()


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
        self.reset_hand()
        


    def grasp_object(self, object_name):
        object = self.og_sim.task.object_scope[object_name]
        primitive_action = self.action_primitives.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, object)
        execute_controller(primitive_action, self.og_sim)
        
    def place_ontop_object(self, object_name):
        object = self.og_sim.task.object_scope[object_name]
        primitive_action = self.action_primitives.apply_ref(StarterSemanticActionPrimitiveSet.PLACE_ON_TOP, object)
        execute_controller(primitive_action, self.og_sim)


    def add_control(self,control):
        self.control_queue.put(control)


    def reach_pose(self, pose, is_local=False):
        robot = self.robot
        pos = th.tensor(pose[:3],device=self.device)
        euler = th.tensor(pose[3:],device=self.device)

        if euler is None:
            quat = T.euler2quat(th.tensor([0,math.pi/2,0], dtype=th.float32)) # 默认上往下抓
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
            # print(joint_trajectory)
            
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
                
    def reach_pose_try_best(self, pose, is_local=False,object_name=None):
        robot = self.robot
        pos = th.tensor(pose[:3],device=self.device)
        euler = th.tensor(pose[3:],device=self.device)

        # if euler is None:
        #     quat = T.euler2quat(th.tensor([0,math.pi/2,0], dtype=th.float32)) # 默认上往下抓
        # else:
        #     quat = T.euler2quat(euler)
        grasp_orientations = [
            T.euler2quat(th.tensor([0, math.pi/2, math.pi/2], dtype=th.float32)),      # 从上往下抓
            # T.euler2quat(th.tensor([0, -math.pi/2, 0], dtype=th.float32)),     # 从下往上抓
            # T.euler2quat(th.tensor([0, 0, 0], dtype=th.float32)),              # 从前往后抓
            # T.euler2quat(th.tensor([0, math.pi, 0], dtype=th.float32)),        # 从后往前抓
            # T.euler2quat(th.tensor([0, 0, math.pi/2], dtype=th.float32)),      # 从左往右抓
            # T.euler2quat(th.tensor([0, 0, -math.pi/2], dtype=th.float32)),     # 从右往左抓
            # T.euler2quat(th.tensor([0, 3*math.pi/4, 0], dtype=th.float32)) # 45度角抓取
        ]
            
            
        # 在pos周围生成更多的目标点
        pos_list = []
        for i in range(20):
            # th.randn()生成的是服从标准正态分布(高斯分布)N(0,1)的随机数
            # 随机生成0.01-0.03之间的扰动幅度
            # scale = (0.01 + 0.02 * th.rand(1, device=self.device))   # rand生成[0,1]之间的随机数
            # 或者直接指定不同方向的最大扰动范围
            # scales = th.tensor([0.004, 0.003, 0.002], device=self.device)  # xyz方向的最大扰动范围不同
            # 每个方向使用不同的scale
            scales = 0. * th.rand(3, device=self.device)  # 生成3个不同的scale值
            pos_list.append(pos + th.randn(3, device=self.device) * scales)  
        # 组合尽可能多的 pos和  grasp_orientations 两两组合得到 pos_sequence和 quat_sequence
        # pos_sequence和 quat_sequence 形状为 [len(grasp_orientations)*10,3] 和 [len(grasp_orientations)*10,4]
        # pos_list 里的每个 pos 都尝试用所有的 grasp_orientations 得到 pos_sequence和 quat_sequence
    
        # 尝试每个位置和方向组合
        for test_pos in pos_list:
            for quat in grasp_orientations:
                try:
                    self.open_gripper()

                    # 创建一个包含两个相同位姿的序列（起始和目标）
                    pos_sequence = th.stack([test_pos, test_pos])  # [2, 3]
                    quat_sequence = th.stack([quat, quat])  # [2, 4]

                    # 如果机器人接近关节限制，则调整关节位置
                    jp = robot.get_joint_positions(normalized=True)
                    if not th.all(th.abs(jp)[:-2] < 0.97):
                        new_jp = jp.clone()
                        new_jp[:-2] = th.clamp(new_jp[:-2], min=-0.95, max=0.95)
                        robot.set_joint_positions(new_jp, normalized=True)
                    og.sim.step()

                    # 计算轨迹
                    successes, paths = self.curobo_mg.compute_trajectories(
                        pos_sequence, 
                        quat_sequence,
                        is_local=is_local,
                        max_attempts=3,  # 减少单次尝试次数
                        timeout=1.0,     # 减少超时时间
                        enable_graph_attempt=2,
                        ik_fail_return=3,
                        enable_finetune_trajopt=True,
                        finetune_attempts=1
                    )

                    # 检查是否有可行解且paths不为None
                    if successes[0] and paths and paths[0] is not None:
                        joint_trajectory = self.curobo_mg.path_to_joint_trajectory(paths[0])
                        if joint_trajectory is not None:
                            
                            # 把当前的 test_pos 在仿真环境中标出来
                            from omni.isaac.core.objects import cuboid
                            # 创建可视化目标点
                            self.visual_cube = cuboid.VisualCuboid(
                                "/World/visual",
                                position=test_pos.cpu().numpy(),  # 转换为numpy数组
                                orientation=np.array([0, 1, 0, 0]),
                                color=np.array([1.0, 0, 0]),  # 红色
                                size=0.01,  # 更小的尺寸，更容易看清
                            )
                            log(f"尝试位置: {test_pos}")  # 打印当前尝试的位置
                            
                            for time_i, joint_positions in enumerate(joint_trajectory):
                                full_action = th.zeros(robot.n_joints, device=joint_positions.device)
                                full_action[4:] = joint_positions
                                
                                # # 控制夹爪
                                if time_i == len(joint_trajectory) - 1:
                                    full_action[-2:] = 0.0  # 关闭夹爪
                                else:
                                    full_action[-2:] = 0.05  # 保持夹爪打开

                                self.og_sim.step(full_action.to('cpu'))

                            self.close_gripper()

                            object_name = self.og_sim.task.load_task_metadata()['inst_to_name'][object_name]
                            # 检测是否抓起了物体
                            # 目前存在问题: curobo碰撞总是发生
                            obj_in_hand = self.action_primitives._get_obj_in_hand()
                            log(f"obj_in_hand: {obj_in_hand}")
                            if obj_in_hand is not None and obj_in_hand.name == object_name:  # 如果有接触
                                log(f"检测到物体接触 {obj_in_hand.name}")
                                return True  # 成功找到并执行了轨迹
                            else:
                                log("未检测到物体接触")
                                continue
                except Exception as e:
                    log(f"尝试位置 {test_pos} 和方向 {quat} 时发生\n错误: {str(e)}")
                    continue
        
        log("所有组合都尝试失败")
        return False  # 所有组合都尝试失败
                

    def reset_hand(self):
        jp = self.get_joint_states()
        self.set_joint_states([
                0.0,
                0.0,  # wheels
                1.0,  # trunk
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
        self.idle_step(10)
        
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

        for i in range(10):
            self.idle_step()
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
        intrinsics = og_utils.get_cam_intrinsics(sensor).reshape(-1)
        # intrinsics = sensor.intrinsic_matrix.reshape(-1).cpu().numpy()
        pose = sensor.get_position_orientation()
        extrinsics = T.pose_inv(T.pose2mat(pose)).reshape(-1).cpu().numpy()

        return {
            'intrinsics': intrinsics,
            'extrinsics': extrinsics
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
            'proprio': proprio_obs
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

    def get_object_pos(self,object_name):
        obj = self.og_sim.task.object_scope[object_name]
        pos = obj.get_position_orientation()[0]
        return {'pos': pos}

    def close_gripper(self):
        self.gripper_control(open=False)
        self.kinematics_config.lock_jointstate.position = th.tensor([0., 0.],device=self.kinematics_config.lock_jointstate.position.device)

    def open_gripper(self):
        self.gripper_control(open=True)
        self.kinematics_config.lock_jointstate.position = th.tensor([0.05, 0.05],device=self.kinematics_config.lock_jointstate.position.device)

    def gripper_control(self, open=True):
        joint_positions = self.robot.get_joint_positions()
        action = th.zeros(self.robot.n_joints)
        action[4] = joint_positions[2]
        action[5] = joint_positions[4]
        action[6:-2] = joint_positions[6:-2]
        gripper_target = 0.05 if open else 0.0
        action[-2:] = gripper_target
        for _ in range(20):
            self.og_sim.step(action.to('cpu'))
        self.gripper_open = open

        self.idle_step(20)


if __name__ == "__main__":
    # print(gm.REMOTE_STREAMING)
    set_logger_entry(__file__)
    
    simulator = Simulator(task_name=None)

    # simulator.load_custom_task('test_task',scene_file_name='scene_file_0')
    from btgym.dataclass.cfg import cfg
    cfg.task_name = "task1"
    cfg.scene_file_name='scene_file_0'
    simulator.load_custom_task(task_name=cfg.task_name, scene_file_name=cfg.scene_file_name)
    
    
    object_name = 'apple.n.01_1'
    simulator.navigate_to_object(object_name=object_name)
    
    
    
    # grasp_pos = [ 0.7103, -3.6875,  0.8163, 0,0,0]
    # 获取物体中心点
    grasp_pos = simulator.get_object_pos(object_name)
    # grasp_pos转为list
    grasp_pos = grasp_pos['pos'].tolist()
    simulator.reach_pose_try_best(grasp_pos,object_name=object_name)
    
    
    # simulator.grasp_object(object_name=object_name)
    # object_name = 'coffee_table.n.01_1'
    # simulator.navigate_to_object(object_name=object_name)
    # simulator.place_ontop_object(object_name=object_name)
    
    

    
    # simulator.load_behavior_task('putting_shoes_on_rack')

    # print(simulator.get_object_pos('shoe_1'))

    # simulator.get_obs()
    # simulator.get_camera_info()
    # pos = [0,1,1]
    # simulator.set_target_visual_pose([*pos,0,0,0])
    # simulator.set_camera_lookat_pos(pos)


    # simulator.load_behavior_task('putting_shoes_on_rack')
    # simulator.get_obs()
    # simulator.get_camera_info()
    # pos = [0,1,1]
    # simulator.set_target_visual_pose([*pos,0,0,0])
    # simulator.set_camera_lookat_pos(pos)

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