import os
import yaml
import omnigibson as og
from omnigibson.macros import gm
import omnigibson.utils.transform_utils as T
import torch as th
import math
import random
import time
import numpy as np
from btgym import ROOT_PATH
from btgym.core.curobo import CuRoboMotionGenerator
from btgym.utils.og_utils import OGCamera
import sys
import importlib.util
import importlib
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
from btgym.utils.logger import log,set_logger_entry
from btgym.dataclass.cfg import cfg
import cv2
import pickle
from omnigibson.utils.constants import semantic_class_id_to_name

th.set_printoptions(precision=4)
code_path = os.path.join(ROOT_PATH, "../examples/vlm_solver/cached")
sys.path.append(code_path)





################################
# 把相机里的点转换到世界坐标系
################################
def convert_points_to_world(points, cam_obs):
    """将图像上的点转换到世界坐标系
    Args:
        points: 图像上的点坐标列表 [(x1,y1), (x2,y2),...]
        cam_obs: 相机观察数据,包含depth和内外参等信息
    Returns:
        world_points: 世界坐标系中的点坐标列表 [(x1,y1,z1), (x2,y2,z2),...]
    """
    world_points = []
    
    # 获取相机内外参
    intrinsics = cam_obs['intrinsics']  # 相机内参矩阵
    extrinsics = cam_obs['extrinsics']  # 相机外参矩阵(相机到世界的变换)
    depth_img = cam_obs['depth']        # 深度图
    
    for point in points:
        x, y = point
        # 获取该点的深度值
        depth = depth_img[int(y), int(x)]
        
        # 图像坐标转相机坐标
        x_cam = (x - intrinsics[0,2]) * depth / intrinsics[0,0]
        y_cam = (y - intrinsics[1,2]) * depth / intrinsics[1,1]
        z_cam = depth
        
        # 相机坐标转世界坐标
        cam_point = np.array([x_cam, y_cam, z_cam, 1.0])
        world_point = extrinsics @ cam_point
        
        world_points.append(world_point[:3])
        
    return world_points



if __name__ == "__main__":
    from grasp_pen import Env
    env = Env()

    points = [(100, 100), (200, 200), (300, 300)]

    # 获取相机观察数据
    cam_obs = env.get_cam_obs()[0]

    # 将图像点转换到世界坐标系
    world_points = convert_points_to_world(points, cam_obs)

    print("世界坐标系中的点位置:")
    for i, point in enumerate(world_points):
        print(f"点 {i+1}: {point}")

    env.idle()