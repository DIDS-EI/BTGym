import math
import torch as th
from omnigibson.utils import transform_utils as T

class Pen:
    def __init__(self, env, obj):
        self.env = env
        self.obj = obj

    # def get_grasp_pose(self):
    #     pos = th.tensor([-0.15, -0.15, 0.72], dtype=th.float32)
    #     euler = th.tensor([0, 90, 90], dtype=th.float32)
    #     pose = (pos, euler)
    #     return pose

    def get_grasp_poses(self, num_poses=10, margin=0.02):
        """
        计算物体可行的抓取姿态。基于物体的轴对齐边界框生成抓取点和方向。
        
        Args:
            num_poses (int): 需要生成的抓取姿态数量
            margin (float): 抓取点距离边界框表面的边距，单位米
            
        Returns:
            list of tuple: 每个元素包含:
                - position (3-array): 抓取点在世界坐标系中的位置 (x,y,z)
                - orientation (4-array): 抓取方向的四元数 (x,y,z,w)
        """
        # 获取物体的边界框信息
        bbox_center_world, bbox_orn_world, bbox_extent, bbox_center_local = self.obj.get_base_aligned_bbox(
            visual=False, xy_aligned=True
        )
        
        grasp_poses = []
        
        # 获取边界框的6个面的中心点作为潜在抓取点
        faces_centers = [
            # 沿x轴正负方向
            (th.tensor([bbox_extent[0]/2, 0, 0]), th.tensor([1, 0, 0])),
            (th.tensor([-bbox_extent[0]/2, 0, 0]), th.tensor([-1, 0, 0])),
            # 沿y轴正负方向
            (th.tensor([0, bbox_extent[1]/2, 0]), th.tensor([0, 1, 0])),
            (th.tensor([0, -bbox_extent[1]/2, 0]), th.tensor([0, -1, 0])),
            # 沿z轴正负方向
            (th.tensor([0, 0, bbox_extent[2]/2]), th.tensor([0, 0, 1])),
            (th.tensor([0, 0, -bbox_extent[2]/2]), th.tensor([0, 0, -1]))
        ]
        
        # 为每个面生成抓取姿态
        poses_per_face = max(1, num_poses // 6)
        for center, normal in faces_centers:
            for _ in range(poses_per_face):
                # 在面的中心周围添加一些随机偏移
                offset = (th.rand(3) - 0.5) * 0.1  # 随机偏移范围为±5cm
                # 确保偏移不会使抓取点超出边界框
                offset = th.clamp(offset, -bbox_extent/2 + margin, bbox_extent/2 - margin)
                
                # 计算抓取点位置（在局部坐标系中）
                grasp_pos_local = center + offset
                
                # 将局部坐标转换为世界坐标
                grasp_pos_world = T.transform_points(
                    grasp_pos_local.unsqueeze(0),
                    T.pose2mat((bbox_center_world, bbox_orn_world))
                ).squeeze(0)
                
                # 计算抓取方向的四元数
                # 使用normal作为抓取方向，需要将局部坐标系的方向转换到世界坐标系
                grasp_orn_world = T.mat2quat(
                    T.quat2mat(bbox_orn_world) @ T.vectors2rotation_matrix(normal, th.tensor([0., 0., 1.]))
                )
                
                grasp_poses.append((grasp_pos_world, grasp_orn_world))
                
                if len(grasp_poses) >= num_poses:
                    return grasp_poses
        
        return grasp_poses[0]