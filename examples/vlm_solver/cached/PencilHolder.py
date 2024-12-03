import math
import torch as th

class PencilHolder:
    def __init__(self, env, obj_name):
        self.env = env
        self.obj_name = obj_name

    def get_pen_release_pose(self, pen):
        pos = th.tensor([-0.32, 0.15, 0.9], dtype=th.float32)
        euler = th.tensor([0, 0, -90], dtype=th.float32)
        pose = (pos, euler)
        return pose