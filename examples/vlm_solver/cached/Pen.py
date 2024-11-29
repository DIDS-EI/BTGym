import torch as th

class Pen:
    def __init__(self, env, obj_name):
        self.env = env
        self.obj_name = obj_name

    def get_grasp_pose(self):
        pos = th.tensor([0.1, -0.15, 0.7163], dtype=th.float32)
        euler = th.tensor([0, 0, 0], dtype=th.float32)
        return pos, euler