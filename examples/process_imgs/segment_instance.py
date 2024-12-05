import os
import cv2
import numpy as np
from btgym import ROOT_PATH
import pickle
from omnigibson.utils.constants import semantic_class_id_to_name
import torch

def process_segmentation(obs, output_dir, cam_id):
    """处理分割图像并保存结果
    
    Args:
        obs: 包含分割图像的观察字典
        output_dir: 输出目录路径
        cam_id: 相机ID
    """
    if 'seg_instance' not in obs or obs['seg_instance'] is None:
        return
        
    seg_instance = obs['seg_instance']
    
    # 转换为numpy数组
    if not isinstance(seg_instance, np.ndarray):
        seg_instance = np.array(seg_instance.cpu())
    if seg_instance.dtype != np.uint8:
        seg_instance = seg_instance.astype(np.uint8)
        
    # 获取唯一的实例ID
    unique_ids = np.unique(seg_instance)
    
    # 创建彩色标记图像
    rgb_img = np.array(obs['rgb'])
    overlay = rgb_img.copy()
    result = rgb_img.copy()  # 创建最终结果图像
    
    # 为每个实例创建彩色掩码
    for i, instance_id in enumerate(unique_ids):
        if instance_id == 0:  # 跳过背景
            continue
            
        # 创建当前实例的掩码
        mask = (seg_instance == instance_id)
        
        # 为每个实例随机生成颜色
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        
        # 应用掩码
        overlay[mask] = color
        
        # 找到轮廓并绘制
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 3)  # 3是轮廓的粗细
        
        # 计算掩码的重心作为标注位置
        y_coords, x_coords = np.where(mask)
        if len(y_coords) > 0:
            center_y = int(np.mean(y_coords))
            center_x = int(np.mean(x_coords))
    
    # 先混合原始图像和标记
    result = cv2.addWeighted(rgb_img, 0.7, overlay, 0.3, 0)
    
    # 在结果图像上重新绘制轮廓
    for i, instance_id in enumerate(unique_ids):
        if instance_id == 0:
            continue
        mask = (seg_instance == instance_id)
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        cv2.drawContours(result, contours, -1, color, 3)
    
    # 最后添加编号标注，确保在最上层
    for i, instance_id in enumerate(unique_ids):
        if instance_id == 0:
            continue
            
        mask = (seg_instance == instance_id)
        y_coords, x_coords = np.where(mask)
        if len(y_coords) > 0:
            center_y = int(np.mean(y_coords))
            center_x = int(np.mean(x_coords))
            
            # 添加黑色描边的白色文字，使文字更清晰
            text = str(i)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            
            # 先画黑色描边
            cv2.putText(result, text, (center_x-2, center_y-2),
                    font, font_scale, (0, 0, 0), thickness+2)
            cv2.putText(result, text, (center_x+2, center_y+2),
                    font, font_scale, (0, 0, 0), thickness+2)
            
            # 再画白色文字
            cv2.putText(result, text, (center_x, center_y),
                    font, font_scale, (255, 255, 255), thickness)
    
    # 保存结果
    seg_path = os.path.join(output_dir, f'camera_{cam_id}_seg_labeled.png')
    cv2.imwrite(seg_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    # 保存原始分割图
    # seg_path = os.path.join(output_dir, f'camera_{cam_id}_seg.png')
    # cv2.imwrite(seg_path, seg_instance)

    return result, seg_instance, len(unique_ids)


def crop_objects_by_ids(selected_ids, rgb_img, seg_instance, output_dir, cam_id, margin=0.1):
    """裁剪指定编号的物体并保存
    
    Args:
        selected_ids (List[int]): 要裁剪的物体编号列表(对应新标注的编号i)
        rgb_img (np.ndarray): RGB图像
        seg_instance (np.ndarray): 分割实例图像
        output_dir (str): 输出目录
        cam_id (int): 相机ID
        margin (float): 裁剪边界外扩的边距比例，默认0.1
    """
    # rgb
    # 确保rgb是numpy数组并且是uint8类型
    # if not isinstance(rgb, np.ndarray):
    #     rgb = np.array(rgb)
    # if rgb.dtype != np.uint8:
    #     rgb = (rgb * 255).astype(np.uint8) 
    
    height, width = rgb_img.shape[:2]
    cropped_images = []
    bboxes = []
    
    # 获取唯一的实例ID(跳过背景0)
    unique_ids = np.unique(seg_instance)
    unique_ids = unique_ids[unique_ids != 0]  # 排除背景0
    
    for obj_id in selected_ids:
        # obj_id 是新标注的编号i
        instance_id = unique_ids[obj_id]  # 直接使用新标注的编号获取对应的实例ID
        
        # 创建该物体的掩码
        mask = (seg_instance == instance_id)
        mask = mask.astype(np.uint8) * 255
        
        # ... (其余裁剪代码保持不变) ...
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            continue
            
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        
        w = x_max - x_min
        h = y_max - y_min
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        x1 = max(0, x_min - margin_x)
        y1 = max(0, y_min - margin_y)
        x2 = min(width, x_max + margin_x)
        y2 = min(height, y_max + margin_y)
        
        cropped = rgb_img[y1:y2, x1:x2].copy()
        
        # 保存时使用新标注的编号
        crop_path = os.path.join(output_dir, f'camera_{cam_id}_object_{obj_id}.png')
        cv2.imwrite(crop_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
        
        cropped_images.append(cropped)
        bboxes.append((x1, y1, x2, y2))
        
        print(f"物体 {obj_id} 已裁剪并保存到: {crop_path}")
        print(f"边界框: ({x1}, {y1}, {x2}, {y2})")
        
    return cropped_images, bboxes


def get_bounding_boxes(obs, cam_id, bbox_type='bbox_2d_tight'):
    """获取场景中物体的边界框
    
    Args:
        cam_id (int): 相机ID
        bbox_type (str): 边界框类型，可选 'bbox_2d_tight', 'bbox_2d_loose', 'bbox_3d'
        
    Returns:
        list: 包含边界框信息的列表，每个元素为字典:
            - 2D边界框: {
                'semantic_id': int,  # 语义ID
                'x_min': int,       # 左上角x坐标
                'y_min': int,       # 左上角y坐标
                'x_max': int,       # 右下角x坐标
                'y_max': int,       # 右下角y坐标
                'occlusion': float  # 遮挡比例
            }
            - 3D边界框: {
                'semantic_id': int,    # 语义ID
                'x_min': float,       # 最小x坐标
                'y_min': float,       # 最小y坐标
                'z_min': float,       # 最小z坐标
                'x_max': float,       # 最大x坐标
                'y_max': float,       # 最大y坐标
                'z_max': float,       # 最大z坐标
                'transform': np.array, # 4x4变换矩阵
                'occlusion': float    # 遮挡比例
            }
    """
    # 获取相机观察数据
    # obs = self.cams[cam_id].get_obs()

    
    if bbox_type not in obs:
        raise ValueError(f"无法获取 {bbox_type} 数据")
    
    bboxes = obs[bbox_type]
    results = []
    
    # 处理边界框数据
    for bbox in bboxes:
        if bbox_type.startswith('bbox_2d'):
            results.append({
                'semantic_id': bbox[0],
                'x_min': bbox[1],
                'y_min': bbox[2],
                'x_max': bbox[3],
                'y_max': bbox[4],
                'occlusion': bbox[5]
            })
        else:  # bbox_3d
            results.append({
                'semantic_id': bbox[0],
                'x_min': bbox[1],
                'y_min': bbox[2],
                'z_min': bbox[3],
                'x_max': bbox[4],
                'y_max': bbox[5],
                'z_max': bbox[6],
                'transform': bbox[7].reshape(4, 4),
                'occlusion': bbox[8]
            })
    
    return results

def visualize_bboxes_2d(cam_id, rgb_img, bboxes, output_path=None):
    """可视化2D边界框
    
    Args:
        cam_id (int): 相机ID
        rgb_img (torch.Tensor or np.ndarray): RGB图像
        bboxes (list): 2D边界框列表
        output_path (str, optional): 输出图像路径
    """
    
    # 确保图像是uint8类型
    if rgb_img.dtype != np.uint8:
        rgb_img = (rgb_img * 255).astype(np.uint8)
    
    # 复制图像以避免修改原图
    img = rgb_img.copy()
    
    # 为每个边界框绘制矩形和标签
    for bbox in bboxes:
        # 获取语义类别名称
        semantic_name = semantic_class_id_to_name().get(int(bbox['semantic_id']), 'unknown')
        
        # 确保坐标是整数
        x_min = int(bbox['x_min'])
        y_min = int(bbox['y_min'])
        x_max = int(bbox['x_max'])
        y_max = int(bbox['y_max'])
        
        # 绘制矩形
        cv2.rectangle(img, 
                     (x_min, y_min), 
                     (x_max, y_max),
                     (0, 255, 0), 2)
        
        # 添加标签
        label = f"{semantic_name} ({bbox['occlusion']:.2f})"
        cv2.putText(img, label, 
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    return img

def visualize_bboxes_3d(obs, cam_id, rgb_img, bboxes, output_path=None):
    """可视化3D边界框
    
    Args:
        obs (dict): 观察数据，包含相机参数
        cam_id (int): 相机ID
        rgb_img (torch.Tensor or np.ndarray): RGB图像
        bboxes (list): 3D边界框列表
        output_path (str, optional): 输出图像路径
    """

    # 将张量转换为numpy数组
    if torch.is_tensor(rgb_img):
        rgb_img = rgb_img.cpu().numpy()
    
    # 确保图像是uint8类型
    if rgb_img.dtype != np.uint8:
        rgb_img = (rgb_img * 255).astype(np.uint8)
    
    # 复制图像以避免修改原图
    img = rgb_img.copy()
    
    # 获取相机参数
    intrinsic = obs['intrinsic']
    extrinsic = obs['extrinsic']
    
    # 为每个3D边界框绘制投影
    for bbox in bboxes:
        # 获取语义类别名称
        semantic_name = semantic_class_id_to_name().get(int(bbox['semantic_id']), 'unknown')
        
        # 创建3D边界框的8个顶点
        x_min, y_min, z_min = bbox['x_min'], bbox['y_min'], bbox['z_min']
        x_max, y_max, z_max = bbox['x_max'], bbox['y_max'], bbox['z_max']
        
        vertices = np.array([
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ])
        
        # 应用变换矩阵
        transform = bbox['transform']
        if torch.is_tensor(transform):
            transform = transform.cpu().numpy()
        vertices = (transform @ np.hstack((vertices, np.ones((8, 1)))).T).T[:, :3]
        
        # 将3D点投影到2D图像平面
        points_2d = point_to_pixel(vertices, intrinsic, extrinsic)
        points_2d = points_2d.astype(np.int32)
        
        # 绘制边界框的边
        edges = [(0,1), (1,2), (2,3), (3,0),
                (4,5), (5,6), (6,7), (7,4),
                (0,4), (1,5), (2,6), (3,7)]
        
        for start, end in edges:
            cv2.line(img, 
                    tuple(points_2d[start]), 
                    tuple(points_2d[end]), 
                    (0, 255, 0), 2)
        
        # 添加标签（使用第一个顶点的位置）
        label = f"{semantic_name} ({bbox['occlusion']:.2f})"
        cv2.putText(img, label, 
                    tuple(points_2d[0]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    return img

def point_to_pixel(pt, intrinsics, extrinsics):
    """将3D点投影到2D像素坐标
    
    Args:
        pt (np.ndarray): (N, 3) 3D点坐标
        intrinsics (torch.Tensor or np.ndarray): (3, 3) 相机内参矩阵
        extrinsics (torch.Tensor or np.ndarray): (4, 4) 相机外参矩阵
        
    Returns:
        np.ndarray: (N, 2) 2D像素坐标
    """
    # 确保所有输入都是numpy数组
    if torch.is_tensor(intrinsics):
        intrinsics = intrinsics.cpu().numpy()
    if torch.is_tensor(extrinsics):
        extrinsics = extrinsics.cpu().numpy()
    if torch.is_tensor(pt):
        pt = pt.cpu().numpy()
    
    # 将3D点转换为齐次坐标
    pt_homo = np.hstack((pt, np.ones((pt.shape[0], 1))))  # (N, 4)
    
    # 将点转换到相机坐标系
    pt_in_cam = extrinsics @ pt_homo.T  # (4, N)
    
    # 由于OmniGibson的坐标系约定，需要对y和z取反
    pt_in_cam[1, :] *= -1
    pt_in_cam[2, :] *= -1
    
    # 取前3个坐标
    pt_in_cam = pt_in_cam[:3, :]  # (3, N)
    
    # 应用相机内参
    pt_in_pixel = intrinsics @ pt_in_cam  # (3, N)
    
    # 归一化齐次坐标
    pt_in_pixel = pt_in_pixel / pt_in_pixel[2, :]  # (3, N)
    
    # 返回像素坐标 (x, y)
    return pt_in_pixel[:2, :].T  # (N, 2)

if __name__ == "__main__":
    cam_id = 0
    folder_path = os.path.join(ROOT_PATH, "../examples/process_imgs")
    obs = pickle.load(open(os.path.join(folder_path, f"camera_{cam_id}_obs.pkl"), "rb"))
    result, seg_instance, seg_num = process_segmentation(obs,folder_path,cam_id)
    
    # 选择输出裁剪的图片
    selected_ids = [2, 3]  #range(0, seg_num)#[2, 3]
    cropped_images, bboxes = crop_objects_by_ids(selected_ids, np.array(obs['rgb']), seg_instance, folder_path, cam_id, margin=0.1)
    
    # 获取并可视化2D边界框
    bboxes = get_bounding_boxes(obs, cam_id, 'bbox_2d_tight')
    bbox_img = visualize_bboxes_2d(cam_id, np.array(obs['rgb']), bboxes, 
                                os.path.join(folder_path, f'camera_{cam_id}_bbox_2d.png'))

    # 获取并可视化3D边界框
    bboxes = get_bounding_boxes(obs, cam_id, 'bbox_3d')
    bbox_img = visualize_bboxes_3d(obs, cam_id, np.array(obs['rgb']), bboxes,
                                os.path.join(folder_path, f'camera_{cam_id}_bbox_3d.png'))
    
