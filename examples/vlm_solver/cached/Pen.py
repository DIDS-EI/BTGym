import math
import torch as th

class Pen:
    def __init__(self, env, obj_name):
        self.env = env
        self.obj_name = obj_name

    def get_grasp_pose(self):
        pos = th.tensor([-0.15, -0.15, 0.72], dtype=th.float32)
        euler = th.tensor([0, 90, 90], dtype=th.float32)
        pose = (pos, euler)
        return pose