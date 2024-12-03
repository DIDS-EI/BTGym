import math
import torch as th
from omnigibson.utils import transform_utils as T

class Pen:
    def __init__(self, env, obj):
        self.env = env
        self.obj = obj

    def get_grasp_pose(self):
        pos = th.tensor([-0.15, -0.15, 0.72], dtype=th.float32)
        euler = th.tensor([0, 90, 90], dtype=th.float32)
        pose = (pos, euler)
        return pose
