"""
UI场景
"""
import sys
import json
import math
from matplotlib import pyplot as plt
# from sklearn.cluster import DBSCAN
import pickle
import time
import os

from btgym.scene.scene import Scene

# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from btgym.utils import get_root_path
root_path = get_root_path()


# from btgym.utils.draw import render_dot_tree


class SceneUI(Scene):
    scene_queue = None
    ui_queue = None
    scene_flag=2
    walker_followed = False
    # camera_interval = 4
    def __init__(self, robot,scene_queue,ui_queue):
        self.scene_queue = scene_queue
        self.ui_queue = ui_queue

        super().__init__(robot)
        self.show_ui = True
        # self.ui_queue.put(('say',"llm_test"))
        self.stoped = False

    def run(self):
        # 基类run
        self._run()
        # 运行并由robot打印每步信息
        while not self.stoped:
            self.step()

    def run_reset(self):
        self.gen_obj()
        pass

    def init_robot(self):
        # init robot

        if self.robot:
            self.robot.set_scene(self)
            self.robot.load_BT()
            self.draw_current_bt()

    def draw_current_bt(self):
        render_dot_tree(self.robot.bt.root,target_directory=self.output_path,name="current_bt",png_only=True)
        self.ui_queue.put(('draw_from_file',"img_view_bt", f"{self.output_path}/current_bt.png"))
        # self.ui_queue.put(('draw_from_file', "img_view_bt", f"{self.output_path}/current_bt.svg"))

    def ui_func(self,args):
        # _,_,output_path = args
        # plt.savefig(output_path)

        self.ui_queue.put(args)

    def _reset(self):
        pass

    def _step(self):
        # print("已运行")
        self.handle_queue_messages()
        # if len(self.sub_task_seq.children) == 0:
        #     question = input("请输入指令：")
        #     if question[-1] == ")":
        #         print(f"设置目标:{question}")
        #         self.new_set_goal(question)
        #     else:
        #         self.customer_say("System",question)
        if self.scene_flag == 1:
            # 如果机器人不在 吧台
            if self.walker_followed:
                return
            end = [self.status.location.X, self.status.location.Y]
            if end[1] >= 600 or end[1] <= 450 or end[0] >= 250:
                # if int(self.status.location.X)!=247 or  int(self.status.location.X)!=520:
                self.walker_followed = True
                self.control_walkers_and_say([[0, False, 150, end[0], end[1], 90, "谢谢！"]])
                self.scene_flag += 1


    def handle_queue_messages(self):
        while not self.scene_queue.empty():
            message = self.scene_queue.get()
            function_name = message[0]
            function = getattr(self, function_name, None)

            args = []
            if len(message)>1:
                args = message[1:]

            result = function(*args)

    def stop(self):
        self.stoped = True

if __name__ == '__main__':
    from btgym.robot.robot import Robot

    robot = Robot()
    ui = UI(Robot)

    # create task
    # task = SceneUI(robot,ui)

