import os
import time
import cv2
import queue
from omnigibson.utils.asset_utils import get_og_scene_path

from btgym.core.ui import UI
from btgym.utils.logger import log
import numpy as np
from btgym.core.simulator import Simulator

# from btgym.core.action_symbolic import action_symbolic_map as action_map
# from btgym.core.action_symbolic import ActionPrimitives

from btgym.core.action_starter import action_starter_map as action_map
from btgym.core.action_starter import ActionPrimitives


from btgym.utils.path import ROOT_PATH
def create_ui(*args):
    ui = UI(*args)
    return ui



class Controller:
    """
    Tiago动作空间为24维，
    0-2 是base的x,y,rz速度
    3-4 是头部的rx,ry 旋转
    5   是上下支架的移动
    6-12 是left_arm的7个关节位置
    13-14 是left_gripper的开合
    15-21 是right_arm的7个关节位置
    22-23 是right_gripper的开合
    """


    def __init__(self, simulator:Simulator):
        self.simulator = simulator

        self.last_control_time = time.time()
        self.control_interval = 1/60

        self.last_step_log_time = time.time()
        self.step_log_interval = 2

        log("Controller step log interval: {}".format(self.step_log_interval))
        
        # self.trav_map = self.get_scene_trav_map()

        self.action_queue = queue.Queue()
        self.doing_task = False
        # self.action_queue.put(Grasp("cologne"))
        # self.action_queue.put(PlaceOnTop("table"))

        self.do_task(self.simulator.current_task_name)
        self.current_action = None

        self.idle_control = self.simulator.idle_control
        self.goal_status_str = ""

    def do_task(self, task_name):
        self.reset()
        self.action_primitives = ActionPrimitives(self.simulator)
        self.doing_task = True
        self.action_list = self.load_plan(f'{ROOT_PATH}/../outputs/bddl_planning/success/{task_name}')
        self.execute_plan(self.action_list)

    def reset(self):
        self.action_queue.queue.clear()
        self.current_action = None

    def load_plan(self, plan_file):
        plan_str = 'loaded plan:'

        action_list = []
        with open(plan_file, 'r') as file:
            plan_lines = file.readlines()[:-1]
        for line in plan_lines:
            line_list = line.strip()[1:-1].split(' ')
            action_list.append(line_list)
            plan_str += f'\n{line_list}'
        log(plan_str)
        return action_list
    

    def execute_plan(self, action_list):
        for action in action_list:
            action_name = action[0]
            action_obj = action[-1]
            self.action_queue.put(action_map[action_name](action_obj))

    def step(self):
        current_time = time.time()
        if current_time - self.last_control_time < self.control_interval:
            return
        self.last_control_time = current_time

        if self.doing_task: 
            self.action_step()


    def action_step(self):
        if self.current_action is None or self.current_action.is_stoped:
            goal_list = self.simulator.og_sim.task.ground_goal_state_options[0]
            goal_num = len(goal_list)
            log_str = f'Goal condition check: {goal_num} goals'
            success_goal_list = []
            for goal in goal_list:
                predicate_name = goal.terms[0]
                predicate_args = goal.terms[1:]
                predicate_str = f'{predicate_name}({", ".join(predicate_args)})'
                predicate_satisfied = goal.currently_satisfied
                log_str += f'\n{predicate_str} -> {predicate_satisfied}'
                if predicate_satisfied:
                    success_goal_list.append(predicate_str)
            log(log_str)
            self.goal_status_str = log_str

            if len(success_goal_list) >= goal_num:
                log('Task completed!')
                self.doing_task = False
                return

            if not self.action_queue.empty():
                self.current_action = self.action_queue.get()
                self.current_action.start(self.action_primitives, self.simulator)
                # self.current_action.step()
                # if self.current_action.is_stoped:
                    # self.action_step()
            else:
                self.simulator.add_control(self.idle_control)
        else:
            self.current_action.step()

        # if not self.action_queue.empty():
        #     action = self.action_queue.get()
        #     action.apply(self.action_primitives, self.simulator)
        #     # self.simulator.add_action(action)

    def check_action_space(self):
        # 获取机器人的action space维度
        action_space = self.simulator.robot.action_space
        action_shape = action_space.shape[0]
        
        # 如果上下限是inf,就设为100/-100
        action_space.high[action_space.high == np.inf] = 100
        action_space.low[action_space.low == -np.inf] = -100
        
        # 每个维度生成12个等间隔值
        samples_per_dim = 12
        steps_per_action = 8
        
        # 遍历每个维度
        for dim in range(action_shape):
            # 生成当前维度的等间隔值
            values = np.linspace(action_space.low[dim], action_space.high[dim], samples_per_dim)[1:-1]
            
            # 对每个值执行动作
            for value in values:
                # 创建动作数组,将当前维度设为指定值,其他维度为0
                action = np.zeros(action_shape)
                action[dim] = value
                
                # 执行steps_per_action次
                for _ in range(steps_per_action):
                    self.action_queue.put(action)
                    


    def log_info(self):
        current_time = time.time()
        if current_time - self.last_step_log_time < self.step_log_interval:
            return
        self.last_step_log_time = current_time

        # log(self.action)


    def get_scene_trav_map(self):
        trav_map = self.simulator.get_trav_map()
        # trav_map = cv2.imread(os.path.join(get_og_scene_path(scene_name), "layout", "floor_trav_no_door_0.png"))
        return trav_map
    

