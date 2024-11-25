import torch
import numpy as np
import json
import os
import argparse
from environment import ReKepOGEnv
from keypoint_proposal import KeypointProposer
from constraint_generation import ConstraintGenerator
from ik_solver import IKSolver
from subgoal_solver import SubgoalSolver
from path_solver import PathSolver
from visualizer import Visualizer
import transform_utils as T
from omnigibson.robots.fetch import Fetch
from utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
)

class Main:
    def __init__(self, scene_file, visualize=False):
        global_config = get_config(config_path="./Rekep/configs/config.yaml")
        self.config = global_config['main']
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.visualize = visualize
        # set random seed
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        # initialize keypoint proposer and constraint generator
        self.keypoint_proposer = KeypointProposer(global_config['keypoint_proposer'])
        self.constraint_generator = ConstraintGenerator(global_config['constraint_generator'])
        # initialize environment
        self.env = ReKepOGEnv(global_config['env'], scene_file, verbose=False)
        # setup ik solver (for reachability cost)
        assert isinstance(self.env.robot, Fetch), "The IK solver assumes the robot is a Fetch robot"
        ik_solver = IKSolver(
            robot_description_path=self.env.robot.robot_arm_descriptor_yamls[self.env.robot.default_arm],
            robot_urdf_path=self.env.robot.urdf_path,
            eef_name=self.env.robot.eef_link_names[self.env.robot.default_arm],
            reset_joint_pos=self.env.reset_joint_pos,
            world2robot_homo=self.env.world2robot_homo,
        )
        # initialize solvers
        self.subgoal_solver = SubgoalSolver(global_config['subgoal_solver'], ik_solver, self.env.reset_joint_pos)
        self.path_solver = PathSolver(global_config['path_solver'], ik_solver, self.env.reset_joint_pos)
        # initialize visualizer
        if self.visualize:
            self.visualizer = Visualizer(global_config['visualizer'], self.env)

    def perform_task(self, instruction, rekep_program_dir=None, disturbance_seq=None):
        self.env.reset()
        cam_obs = self.env.get_cam_obs()
        rgb = cam_obs[self.config['vlm_camera']]['rgb']
        points = cam_obs[self.config['vlm_camera']]['points']
        mask = cam_obs[self.config['vlm_camera']]['seg']
        
        
        # 调用原子动作
        
        # 
        
        
        
        
        # # ====================================
        # # = keypoint proposal and constraint generation 关键点提议和约束生成
        # # ====================================
        # if rekep_program_dir is None:
        #     # 调用大模型获取关键点，KeypointProposer 可能使用视觉语言模型(VLM)来分析场景图像，识别关键物体和位置。
        #     keypoints, projected_img = self.keypoint_proposer.get_keypoints(rgb, points, mask)
        #     print(f'{bcolors.HEADER}Got {len(keypoints)} proposed keypoints{bcolors.ENDC}')
        #     # 显示包含关键点的2D图像
        #     if self.visualize:
        #         self.visualizer.show_img(projected_img)
        #     metadata = {'init_keypoint_positions': keypoints, 'num_keypoints': len(keypoints)}
        #     # 调用大模型生成约束，ConstraintGenerator 可能使用优化方法来生成一系列约束，以确保生成的路径是有效的。
        #     rekep_program_dir = self.constraint_generator.generate(projected_img, instruction, metadata)
        #     print(f'{bcolors.HEADER}Constraints generated{bcolors.ENDC}')
        # # ====================================
        # # = execute
        # # ====================================
        # self._execute(rekep_program_dir, disturbance_seq)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='pen', help='task to perform')
    parser.add_argument('--use_cached_query', action='store_true', help='instead of querying the VLM, use the cached query')
    parser.add_argument('--apply_disturbance', action='store_true', help='apply disturbance to test the robustness')
    parser.add_argument('--visualize', action='store_true', help='visualize each solution before executing (NOTE: this is blocking and needs to press "ESC" to continue)')
    args = parser.parse_args()

    if args.apply_disturbance:
        assert args.task == 'pen' and args.use_cached_query, 'disturbance sequence is only defined for cached scenario'

    # ====================================
    # = pen task disturbance sequence
    # ====================================

    task_list = {
        'pen': {
            'scene_file': './Rekep/configs/og_scene_file_ruler.json',
            'instruction': 'reorient the red pen and drop it upright into the black pen holder',
            'rekep_program_dir': './vlm_query/pen',
            'disturbance_seq': None,
            # 'disturbance_seq': {1: stage1_disturbance_seq, 2: stage2_disturbance_seq, 3: stage3_disturbance_seq},
            },
    }
    task = task_list['pen']
    scene_file = task['scene_file']
    instruction = task['instruction']
    main = Main(scene_file, visualize=args.visualize)
    main.perform_task(instruction,
                    rekep_program_dir=task['rekep_program_dir'] if args.use_cached_query else None,
                    disturbance_seq=task.get('disturbance_seq', None) if args.apply_disturbance else None)