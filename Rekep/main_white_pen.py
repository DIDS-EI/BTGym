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

os.chdir(os.path.dirname(__file__))


class Main:
    def __init__(self, scene_file, visualize=False):
        global_config = get_config(config_path="./configs/config.yaml")
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
        # ====================================
        # = keypoint proposal and constraint generation 关键点提议和约束生成
        # ====================================
        if rekep_program_dir is None:
            # 调用视觉模型获取关键点，KeypointProposer 可能使用视觉语言模型(VLM)来分析场景图像，识别关键物体和位置。
            keypoints, projected_img = self.keypoint_proposer.get_keypoints(rgb, points, mask)
            print(f'{bcolors.HEADER}Got {len(keypoints)} proposed keypoints{bcolors.ENDC}')
            # 显示包含关键点的2D图像
            if self.visualize:
                self.visualizer.show_img(projected_img)
            metadata = {'init_keypoint_positions': keypoints, 'num_keypoints': len(keypoints)}
            # 调用大模型生成约束，ConstraintGenerator 可能使用优化方法来生成一系列约束，以确保生成的路径是有效的。
            rekep_program_dir = self.constraint_generator.generate(projected_img, instruction, metadata)
            print(f'{bcolors.HEADER}Constraints generated{bcolors.ENDC}')
        # ====================================
        # = execute
        # ====================================
        self._execute(rekep_program_dir, disturbance_seq)

    def _update_disturbance_seq(self, stage, disturbance_seq):
        if disturbance_seq is not None:
            if stage in disturbance_seq and not self.applied_disturbance[stage]:
                # set the disturbance sequence, the generator will yield and instantiate one disturbance function for each env.step until it is exhausted
                self.env.disturbance_seq = disturbance_seq[stage](self.env)
                self.applied_disturbance[stage] = True

    def _execute(self, rekep_program_dir, disturbance_seq=None):
        # load metadata 加载元数据
        with open(os.path.join(rekep_program_dir, 'metadata.json'), 'r') as f:
            self.program_info = json.load(f)
        self.applied_disturbance = {stage: False for stage in range(1, self.program_info['num_stages'] + 1)}
        # register keypoints to be tracked 注册关键点
        self.env.register_keypoints(self.program_info['init_keypoint_positions'])
        # load constraints 加载约束
        self.constraint_fns = dict()
        for stage in range(1, self.program_info['num_stages'] + 1):  # stage starts with 1
            stage_dict = dict()
            for constraint_type in ['subgoal', 'path']:
                load_path = os.path.join(rekep_program_dir, f'stage{stage}_{constraint_type}_constraints.txt')
                get_grasping_cost_fn = get_callable_grasping_cost_fn(self.env)  # special grasping function for VLM to call
                stage_dict[constraint_type] = load_functions_from_txt(load_path, get_grasping_cost_fn) if os.path.exists(load_path) else []
            self.constraint_fns[stage] = stage_dict
        
        # bookkeeping of which keypoints can be moved in the optimization
        # 记录优化中哪些关键点可以移动
        self.keypoint_movable_mask = np.zeros(self.program_info['num_keypoints'] + 1, dtype=bool)
        self.keypoint_movable_mask[0] = True  # first keypoint is always the ee, so it's movable

        # main loop
        self.last_sim_step_counter = -np.inf
        self._update_stage(1)
        while True:
            scene_keypoints = self.env.get_keypoint_positions()
            self.keypoints = np.concatenate([[self.env.get_ee_pos()], scene_keypoints], axis=0)  # first keypoint is always the ee
            self.curr_ee_pose = self.env.get_ee_pose()
            self.curr_joint_pos = self.env.get_arm_joint_postions()
            self.sdf_voxels = self.env.get_sdf_voxels(self.config['sdf_voxel_size'])
            self.collision_points = self.env.get_collision_points()
            # ====================================
            # = decide whether to backtrack 决定是否回退
            # ====================================
            backtrack = False
            if self.stage > 1:
                path_constraints = self.constraint_fns[self.stage]['path']
                for constraints in path_constraints:
                    violation = constraints(self.keypoints[0], self.keypoints[1:])
                    if violation > self.config['constraint_tolerance']:
                        backtrack = True
                        break
            if backtrack:
                # determine which stage to backtrack to based on constraints 根据约束确定回退到哪个阶段
                for new_stage in range(self.stage - 1, 0, -1):
                    path_constraints = self.constraint_fns[new_stage]['path']
                    # if no constraints, we can safely backtrack 如果没有约束，我们可以安全地回退
                    if len(path_constraints) == 0:
                        break
                    # otherwise, check if all constraints are satisfied 否则，检查所有约束是否满足
                    all_constraints_satisfied = True
                    for constraints in path_constraints:
                        violation = constraints(self.keypoints[0], self.keypoints[1:])
                        if violation > self.config['constraint_tolerance']:
                            all_constraints_satisfied = False
                            break
                    if all_constraints_satisfied:   
                        break
                print(f"{bcolors.HEADER}[stage={self.stage}] backtrack to stage {new_stage}{bcolors.ENDC}")
                self._update_stage(new_stage)
            else:
                # apply disturbance
                self._update_disturbance_seq(self.stage, disturbance_seq)
                # ====================================
                # = get optimized plan
                # ====================================
                if self.last_sim_step_counter == self.env.step_counter:
                    print(f"{bcolors.WARNING}sim did not step forward within last iteration (HINT: adjust action_steps_per_iter to be larger or the pos_threshold to be smaller){bcolors.ENDC}")
                next_subgoal = self._get_next_subgoal(from_scratch=self.first_iter)
                next_path = self._get_next_path(next_subgoal, from_scratch=self.first_iter)
                self.first_iter = False
                self.action_queue = next_path.tolist()
                self.last_sim_step_counter = self.env.step_counter

                # ====================================
                # = execute 执行
                # ====================================
                # determine if we proceed to the next stage 确定是否进入下一个阶段
                count = 0
                
                # while len(self.action_queue) > 0 and count < 30: #self.config['action_steps_per_iter']
                #     next_action = self.action_queue.pop(0)
                #     precise = len(self.action_queue) == 0
                #     self.env.execute_action(next_action, precise=precise,use_curobo=use_curobo)
                #     count += 1
                
                # CuRobo模式: 一次性执行整个轨迹
                precise = len(self.action_queue) == 0
                next_action = self.action_queue  # 整个轨迹序列
                self.env.execute_action(next_action, precise=precise, use_curobo=True)
                self.action_queue = []  # 清空队列
                count = len(self.action_queue)  # 结束循环
                
                if len(self.action_queue) == 0:
                    if self.is_grasp_stage:
                        self._execute_grasp_action()
                    elif self.is_release_stage:
                        self._execute_release_action()
                    # if completed, save video and return 如果完成，保存视频并返回
                    if self.stage == self.program_info['num_stages']: 
                        self.env.sleep(2.0)
                        save_path = self.env.save_video()
                        print(f"{bcolors.OKGREEN}Video saved to {save_path}\n\n{bcolors.ENDC}")
                        return
                    # progress to next stage 进入下一个阶段
                    self._update_stage(self.stage + 1)

    def _get_next_subgoal(self, from_scratch):
        subgoal_constraints = self.constraint_fns[self.stage]['subgoal']
        path_constraints = self.constraint_fns[self.stage]['path']
        subgoal_pose, debug_dict = self.subgoal_solver.solve(self.curr_ee_pose,
                                                            self.keypoints,
                                                            self.keypoint_movable_mask,
                                                            subgoal_constraints,
                                                            path_constraints,
                                                            self.sdf_voxels,
                                                            self.collision_points,
                                                            self.is_grasp_stage,
                                                            self.curr_joint_pos,
                                                            from_scratch=from_scratch)
        subgoal_pose_homo = T.convert_pose_quat2mat(subgoal_pose)
        # if grasp stage, back up a bit to leave room for grasping
        if self.is_grasp_stage:
            subgoal_pose[:3] += subgoal_pose_homo[:3, :3] @ np.array([-self.config['grasp_depth'] / 2.0, 0, 0])
        debug_dict['stage'] = self.stage
        print_opt_debug_dict(debug_dict)
        # 显示3D点云
        if self.visualize:
            self.visualizer.visualize_subgoal(subgoal_pose)
        return subgoal_pose

    
    # def _get_next_path2(self, next_subgoal, from_scratch):
    #     path_constraints = self.constraint_fns[self.stage]['path']
    #     path, debug_dict = self.path_solver.solve(self.curr_ee_pose,
    #                                                 next_subgoal,
    #                                                 self.keypoints,
    #                                                 self.keypoint_movable_mask,
    #                                                 path_constraints,
    #                                                 self.sdf_voxels,
    #                                                 self.collision_points,
    #                                                 self.curr_joint_pos,
    #                                                 from_scratch=from_scratch)
    #     # 打印调试信息
    #     # path shape 是 [2, 7]
    #     # array([[-0.31993555, -0.14992112,  0.83190123,  0.47098515,  0.52738054,
    #     #     -0.47304245,  0.52561734],
    #     #    [-0.32717696, -0.09191327,  0.75094692,  0.46870999,  0.52146103,
    #     #     -0.4760604 ,  0.53080678]])
        
    #     print_opt_debug_dict(debug_dict)
    #     processed_path = self._process_path(path) 
    #     # processed_path shape 是 [5, 8]
    #     # array([[-0.31508186, -0.30122894,  0.81732434,  0.46871074,  0.52146082,
    #     #         -0.47605978,  0.53080688,  0.        ],
    #     #     [-0.31618748, -0.22934496,  0.84355521,  0.47000522,  0.52482328,
    #     #         -0.47435278,  0.52786841,  0.        ],
    #     #     [-0.31857187, -0.17049769,  0.84123593,  0.47100263,  0.52742602,
    #     #         -0.47301897,  0.52557718,  0.        ],
    #     #     [-0.32223503, -0.12468713,  0.8103665 ,  0.47036806,  0.52576981,
    #     #         -0.47386929,  0.5270369 ,  0.        ],
    #     #     [-0.32717696, -0.09191327,  0.75094692,  0.46870999,  0.52146103,
    #     #         -0.4760604 ,  0.53080678,  0.        ]])
    #     if self.visualize:
    #         self.visualizer.visualize_path(processed_path)
    #     return processed_path

    def _get_next_path(self, next_subgoal, from_scratch):
        # 使用 curobo 实现
        # pass
        # 初始化运动规划器
        # subgoal_pose = [x, y, z, qx, qy, qz, qw]  # shape: (7,)
        """使用 CuRobo 计算从当前位置到子目标的路径
        Args:
            next_subgoal (np.ndarray): 目标姿态 [x,y,z,qx,qy,qz,qw]
            from_scratch (bool): 是否从头开始规划
            
        Returns:
            np.ndarray: 路径点序列,每个点包含位置和姿态 [N, 8] (包含夹爪动作)
        """
        from btgym import ROOT_PATH
        from btgym.core.curobo import CuRoboMotionGenerator
        # 初始化 CuRobo 运动规划器(如果还没有初始化)
        if not hasattr(self, 'curobo_mg'):
            self.curobo_mg = CuRoboMotionGenerator(
                self.env.robot,
                robot_cfg_path=os.path.join(ROOT_PATH, "assets/fetch_description_curobo.yaml"),
                debug=False
            )
            # 设置夹爪初始位置
            self.curobo_mg.mg.kinematics.lock_joints = {
                "r_gripper_finger_joint": 0.0,
                "l_gripper_finger_joint": 0.0
            }
        target_pos = torch.tensor(next_subgoal[:3], dtype=torch.float32)
        target_quat = torch.tensor(next_subgoal[3:], dtype=torch.float32)
        # 构建位置和姿态序列
        pos_sequence = torch.stack([target_pos, target_pos])
        quat_sequence = torch.stack([target_quat, target_quat])
        # 检查是否有物体被抓取
        attached_obj = None
        if self.env.is_grasping():
            # 获取被抓取的物体
            for obj in self.env.og_env.scene.objects:
                if self.env.is_grasping(obj):
                    attached_obj = obj
                    break

        # 计算轨迹
        successes, paths = self.curobo_mg.compute_trajectories(
            pos_sequence, 
            quat_sequence,
            attached_obj=attached_obj
        )
        if successes[0]:
            # 获取路径点
            joint_positions = paths[0].position.cpu().numpy()  # [N, 8] numpy array
            
            # joint_positions 中的 N 太大，删减一些
            desired_steps = 5  # 期望的轨迹点数
            step = max(len(joint_positions) // desired_steps, 1)  # 计算采样步长
            joint_positions = joint_positions[::step]  # 等间隔采样 
            
            
            
            # 获取当前末端执行器位姿和目标位姿
            start_pose = self.env.get_ee_pose()  # 当前末端执行器位姿
            end_pose = next_subgoal  # 目标位姿
            
            # 构造用于可视化的位姿序列
            num_steps = len(joint_positions)
            ee_action_seq = np.zeros((num_steps, 8))
            
            # 在起点和终点之间进行线性插值
            for i in range(num_steps):
                t = i / (num_steps - 1)  # 插值参数 [0, 1]
                # 位置线性插值
                ee_action_seq[i, :3] = (1 - t) * start_pose[:3] + t * end_pose[:3]
                # 姿态球面线性插值 (SLERP)
                ee_action_seq[i, 3:7] = T.quat_slerp(start_pose[3:], end_pose[3:], t)
                # 夹爪动作
                ee_action_seq[i, 7] = self.env.get_gripper_null_action()
            
            # 可视化路径
            if self.visualize:
                self.visualizer.visualize_path(ee_action_seq)
            return ee_action_seq
        else:
            raise RuntimeError("Failed to find valid path")  
            # 获取路径点
            # path_poses = paths[0]  # [N, 7] 位置和姿态
            
            # 添加夹爪动作
            # gripper_actions = torch.full((path_poses.shape[0], 1), 
            #                           self.env.get_gripper_null_action())
            # path = torch.cat([path_poses, gripper_actions], dim=1)  # [N, 8]
            
            # 可视化路径(如果需要)
            # if self.visualize:
            #     self.visualizer.visualize_path(path.numpy())
                
            # return path.numpy()
            
        
 

    # def _process_path(self, path):
    #     # spline interpolate the path from the current ee pose
    #     full_control_points = np.concatenate([
    #         self.curr_ee_pose.reshape(1, -1),
    #         path,
    #     ], axis=0)
    #     num_steps = get_linear_interpolation_steps(full_control_points[0], full_control_points[-1],
    #                                                 self.config['interpolate_pos_step_size'],
    #                                                 self.config['interpolate_rot_step_size'])
    #     dense_path = spline_interpolate_poses(full_control_points, num_steps)
    #     # add gripper action
    #     ee_action_seq = np.zeros((dense_path.shape[0], 8))
    #     ee_action_seq[:, :7] = dense_path
    #     ee_action_seq[:, 7] = self.env.get_gripper_null_action()
    #     return ee_action_seq # shape 为 （5, 8）




    def _update_stage(self, stage):
        # update stage
        self.stage = stage
        self.is_grasp_stage = self.program_info['grasp_keypoints'][self.stage - 1] != -1
        self.is_release_stage = self.program_info['release_keypoints'][self.stage - 1] != -1
        # can only be grasp stage or release stage or none
        assert self.is_grasp_stage + self.is_release_stage <= 1, "Cannot be both grasp and release stage"
        if self.is_grasp_stage:  # ensure gripper is open for grasping stage
            self.env.open_gripper()
        # clear action queue
        self.action_queue = []
        # update keypoint movable mask
        self._update_keypoint_movable_mask()
        self.first_iter = True

    def _update_keypoint_movable_mask(self):
        for i in range(1, len(self.keypoint_movable_mask)):  # first keypoint is ee so always movable
            keypoint_object = self.env.get_object_by_keypoint(i - 1)
            self.keypoint_movable_mask[i] = self.env.is_grasping(keypoint_object)

    def _execute_grasp_action(self):
        pregrasp_pose = self.env.get_ee_pose()
        grasp_pose = pregrasp_pose.copy()
        grasp_pose[:3] += T.quat2mat(pregrasp_pose[3:]) @ np.array([self.config['grasp_depth'], 0, 0])
        grasp_action = np.concatenate([grasp_pose, [self.env.get_gripper_close_action()]])
        self.env.execute_action(grasp_action, precise=True,use_curobo=use_curobo)
    
    def _execute_release_action(self):
        self.env.open_gripper()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='pen', help='task to perform')
    parser.add_argument('--use_cached_query', default=True, action='store_true', help='instead of querying the VLM, use the cached query')
    parser.add_argument('--apply_disturbance', action='store_true', help='apply disturbance to test the robustness')
    parser.add_argument('--visualize', default=True, action='store_true', help='visualize each solution before executing (NOTE: this is blocking and needs to press "ESC" to continue)')
    args = parser.parse_args()

    if args.apply_disturbance:
        assert args.task == 'pen' and args.use_cached_query, 'disturbance sequence is only defined for cached scenario'

    
    use_curobo = True
    
    
    
    # ====================================
    # = pen task disturbance sequence
    # ====================================
    def stage1_disturbance_seq(env):
        """
        Move the pen in stage 0 when robot is trying to grasp the pen
        """
        pen = env.og_env.scene.object_registry("name", "pen_1")
        holder = env.og_env.scene.object_registry("name", "pencil_holder_1")
        # disturbance sequence
        pos0, orn0 = pen.get_position_orientation()
        pose0 = np.concatenate([pos0, orn0])
        pos1 = pos0 + np.array([-0.08, 0.0, 0.0])
        orn1 = T.quat_multiply(T.euler2quat(np.array([0, 0, np.pi/4])), orn0)
        pose1 = np.concatenate([pos1, orn1])
        pos2 = pos1 + np.array([0.10, 0.0, 0.0])
        orn2 = T.quat_multiply(T.euler2quat(np.array([0, 0, -np.pi/2])), orn1)
        pose2 = np.concatenate([pos2, orn2])
        control_points = np.array([pose0, pose1, pose2])
        pose_seq = spline_interpolate_poses(control_points, num_steps=25)
        def disturbance(counter):
            if counter < len(pose_seq):
                pose = pose_seq[counter]
                pos, orn = pose[:3], pose[3:]
                pen.set_position_orientation(pos, orn)
                counter += 1
        counter = 0
        while True:
            yield disturbance(counter)
            counter += 1
    
    def stage2_disturbance_seq(env):
        """
        Take the pen out of the gripper in stage 1 when robot is trying to reorient the pen
        """
        apply_disturbance = env.is_grasping()
        pen = env.og_env.scene.object_registry("name", "pen_1")
        holder = env.og_env.scene.object_registry("name", "pencil_holder_1")
        # disturbance sequence
        pos0, orn0 = pen.get_position_orientation()
        pose0 = np.concatenate([pos0, orn0])
        pose1 = np.array([-0.30, -0.15, 0.71, -0.7071068, 0, 0, 0.7071068])
        control_points = np.array([pose0, pose1])
        pose_seq = spline_interpolate_poses(control_points, num_steps=25)
        def disturbance(counter):
            if apply_disturbance:
                if counter < 20:
                    if counter > 15:
                        env.robot.release_grasp_immediately()  # force robot to release the pen
                    else:
                        pass  # do nothing for the other steps
                elif counter < len(pose_seq) + 20:
                    env.robot.release_grasp_immediately()  # force robot to release the pen
                    pose = pose_seq[counter - 20]
                    pos, orn = pose[:3], pose[3:]
                    pen.set_position_orientation(pos, orn)
                    counter += 1
        counter = 0
        while True:
            yield disturbance(counter)
            counter += 1
    
    def stage3_disturbance_seq(env):
        """
        Move the holder in stage 2 when robot is trying to drop the pen into the holder
        """
        pen = env.og_env.scene.object_registry("name", "pen_1")
        holder = env.og_env.scene.object_registry("name", "pencil_holder_1")
        # disturbance sequence
        pos0, orn0 = holder.get_position_orientation()
        pose0 = np.concatenate([pos0, orn0])
        pos1 = pos0 + np.array([-0.02, -0.15, 0.0])
        orn1 = orn0
        pose1 = np.concatenate([pos1, orn1])
        control_points = np.array([pose0, pose1])
        pose_seq = spline_interpolate_poses(control_points, num_steps=5)
        def disturbance(counter):
            if counter < len(pose_seq):
                pose = pose_seq[counter]
                pos, orn = pose[:3], pose[3:]
                holder.set_position_orientation(pos, orn)
                counter += 1
        counter = 0
        while True:
            yield disturbance(counter)
            counter += 1

    task_list = {
        # 'pen': {
        # 'scene_file': './configs/og_scene_file_red_pen.json',
        # 'instruction': 'reorient the red pen and drop it upright into the black pen holder',
        # 'rekep_program_dir': './vlm_query/pen',
        # 'disturbance_seq': {1: stage1_disturbance_seq, 2: stage2_disturbance_seq, 3: stage3_disturbance_seq},
        # },
                
        'pen': {
            'scene_file': './configs/og_scene_file_pen.json',
            'instruction': 'reorient the white pen and drop it upright into the black pen holder',
            'rekep_program_dir': './vlm_query/pen',
            'disturbance_seq': {1: stage1_disturbance_seq, 2: stage2_disturbance_seq, 3: stage3_disturbance_seq},
            },
        
        
        # 'pen': {
        #     'scene_file': './configs/og_scene_file_red_pen.json',
        #     'instruction': 'reorient the red pen and drop it upright into the black pen holder',
        #     'rekep_program_dir': './vlm_query/2024-10-14_08-44-42_reorient_the_red_pen_and_drop_it_upright_into_the_black_pen_holder',# './vlm_query/pen',
        #     'disturbance_seq': None,
        #     'disturbance_seq': {1: stage1_disturbance_seq, 2: stage2_disturbance_seq, 3: stage3_disturbance_seq},
        #     },
        
        
        # 'bottle_of_cologne': {
        #     'scene_file': './configs/og_scene_file_bottle_of_cologne.json',
        #     'instruction': 'reorient the bottle_of_cologne and drop it upright into the black pen holder',
        #     'rekep_program_dir': './vlm_query/bottle_of_cologne',
        #     'disturbance_seq': None,
        #     # 'disturbance_seq': {1: stage1_disturbance_seq, 2: stage2_disturbance_seq, 3: stage3_disturbance_seq},
        #     },
        # 'cologne_to_table': {
        #     'scene_file': './configs/og_scene_file_cologne_to_table.json',
        #     'instruction': 'move the bottle_of_cologne to the other table on your right',
        #     'rekep_program_dir': './vlm_query/cologne_to_table',
        #     'disturbance_seq': None,
        #     # 'disturbance_seq': {1: stage1_disturbance_seq, 2: stage2_disturbance_seq, 3: stage3_disturbance_seq},
        #     },
    }
    task = task_list['pen']
    scene_file = task['scene_file']
    instruction = task['instruction']
    main = Main(scene_file, visualize=args.visualize)
    main.perform_task(instruction,
                    rekep_program_dir=task['rekep_program_dir'] if args.use_cached_query else None,
                    disturbance_seq=task.get('disturbance_seq', None) if args.apply_disturbance else None)
    
    
    
