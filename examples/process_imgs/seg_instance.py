import os
import cv2
import numpy as np
from btgym import ROOT_PATH

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
    seg_path = os.path.join(output_dir, f'camera_{cam_id}_seg.png')
    cv2.imwrite(seg_path, seg_instance)

    return result, seg_instance

if __name__ == "__main__":
    folder_path = os.path.join(ROOT_PATH, "../examples/process_imgs")
    img = cv2.imread(os.path.join(folder_path, "camera_0_rgb.png"))
    process_segmentation(img,folder_path,0)