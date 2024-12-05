import os
import cv2
import numpy as np

folder_path = os.path.dirname(os.path.abspath(__file__))

def draw_coordinate_system(img, scale=10, margin=50):
    """绘制带箭头和刻度的坐标系,原点在左下角
    
    Args:
        img: 输入图像
        scale: 基础刻度间隔(像素)
        margin: 边距大小(像素)
    """
    if img is None:
        raise ValueError("Input image cannot be None")
        
    height, width = img.shape[:2]
    
    # 创建带黑色边框的大图
    new_height = height + margin * 2
    new_width = width + margin * 2
    new_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    # 将原图复制到中心位置
    new_img[margin:margin+height, margin:margin+width] = img
    
    # 设置原点在左下角
    origin = (margin, new_height - margin)
    
    # 绘制细网格线(每个scale一条)
    for x in range(origin[0], new_width-margin, scale):
        color = (240, 240, 240) if (x-origin[0]) % (scale*10) != 0 else (200, 200, 200)
        thickness = 1 if (x-origin[0]) % (scale*10) != 0 else 2
        cv2.line(new_img, (x, margin), (x, origin[1]), color, thickness)
    
    for y in range(margin, origin[1], scale):
        color = (240, 240, 240) if (origin[1]-y) % (scale*10) != 0 else (200, 200, 200)
        thickness = 1 if (origin[1]-y) % (scale*10) != 0 else 2
        cv2.line(new_img, (origin[0], y), (new_width-margin, y), color, thickness)
    
    # 绘制坐标轴(最粗线)
    # X轴
    cv2.line(new_img, origin, (new_width-margin, origin[1]), (255, 255, 255), 2)
    # X轴箭头
    cv2.arrowedLine(new_img, (new_width-margin-40, origin[1]), 
                    (new_width-margin, origin[1]), (255, 255, 255), 2, tipLength=0.05)
    # Y轴
    cv2.line(new_img, origin, (origin[0], margin), (255, 255, 255), 2)
    # Y轴箭头
    cv2.arrowedLine(new_img, (origin[0], margin+40), (origin[0], margin), 
                    (255, 255, 255), 2, tipLength=0.05)
    
    # 添加坐标轴标签
    cv2.putText(new_img, "X", (new_width-margin-30, origin[1]+30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(new_img, "Y", (origin[0]-30, margin+30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 添加刻度和数字(每10个刻度标注一次)
    # X轴刻度
    for i in range(0, int((width)/(scale*10)) + 1):
        x = origin[0] + i*scale*10
        if x >= origin[0] and x < new_width-margin:
            cv2.line(new_img, (x, origin[1]-5), (x, origin[1]+5), (255, 255, 255), 2)
            if i != 0:  # 跳过原点
                cv2.putText(new_img, str(i), (x-10, origin[1]+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Y轴刻度
    for i in range(0, int((height)/(scale*10)) + 1):
        y = origin[1] - i*scale*10
        if y >= margin and y <= origin[1]:
            cv2.line(new_img, (origin[0]-5, y), (origin[0]+5, y), (255, 255, 255), 2)
            if i != 0:  # 跳过原点
                cv2.putText(new_img, str(i), (origin[0]-25, y+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 标记原点
    cv2.putText(new_img, "O", (origin[0]-20, origin[1]+20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return new_img

if __name__ == "__main__":
    img = cv2.imread(os.path.join(folder_path, "camera_0_rgb.png"))
    # 创建坐标系
    img = draw_coordinate_system(img)
    
    # 保存图像
    cv2.imwrite(os.path.join(folder_path, "camera_0_rgb_grids.png"), img)
