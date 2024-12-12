'''
数据格式：
输入：
图片RGBD，

输出：
操作点在RGBD中的坐标，

对于抓取任务：
要抓取的物体，抓取点的坐标和方向
要移动的物体，移动点的坐标和方向
'''



import numpy as np
import os
import cv2
from btgym import ROOT_PATH
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from btgym.llm.llm import LLM
import pickle
def visualize_poses(global_pose, robot_frame_pose):
    """同时可视化全局坐标系和机器人坐标系下的位姿"""
    fig = plt.figure(figsize=(15, 7))
    
    # 设置坐标轴和箭头的长度
    axis_length = 0.08  # 减小坐标轴长度
    
    # 全局坐标系下的位姿
    ax1 = fig.add_subplot(121, projection='3d')
    position = global_pose[:3]
    orientation = global_pose[3:]
    
    # 绘制位置点
    ax1.scatter(position[0], position[1], position[2], c='r', marker='o', s=100, label='Position')
    
    # 从四元数计算旋转矩阵并显示坐标轴
    from scipy.spatial.transform import Rotation
    r = Rotation.from_quat([orientation[0], orientation[1], orientation[2], orientation[3]])
    rotation_matrix = r.as_matrix()
    
    # 绘制坐标轴
    colors = ['r', 'g', 'b']
    for i in range(3):
        direction = rotation_matrix[:, i]
        ax1.quiver(position[0], position[1], position[2],
                  direction[0], direction[1], direction[2],
                  length=axis_length, color=colors[i],
                  label=f'{"XYZ"[i]}-axis')
    
    # 设置坐标轴范围
    range_size = 0.2  # 减小视图范围
    ax1.set_xlim([position[0] - range_size, position[0] + range_size])
    ax1.set_ylim([position[1] - range_size, position[1] + range_size])
    ax1.set_zlim([position[2] - range_size, position[2] + range_size])
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Global Frame Subgoal')
    ax1.legend()
    
    # 机器人坐标系下的位姿
    ax2 = fig.add_subplot(122, projection='3d')
    position = robot_frame_pose[:3]
    orientation = robot_frame_pose[3:]
    
    # 绘制位置点
    ax2.scatter(position[0], position[1], position[2], c='r', marker='o', s=100, label='Position')
    
    # 计算旋转矩阵
    r = Rotation.from_quat([orientation[0], orientation[1], orientation[2], orientation[3]])
    rotation_matrix = r.as_matrix()
    
    # 绘制坐标轴
    for i in range(3):
        direction = rotation_matrix[:, i]
        ax2.quiver(position[0], position[1], position[2],
                  direction[0], direction[1], direction[2],
                  length=axis_length, color=colors[i],
                  label=f'{"XYZ"[i]}-axis')
    
    # 设置坐标轴范围
    ax2.set_xlim([position[0] - range_size, position[0] + range_size])
    ax2.set_ylim([position[1] - range_size, position[1] + range_size])
    ax2.set_zlim([position[2] - range_size, position[2] + range_size])
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Robot Frame Subgoal')
    ax2.legend()
    
    # 设置相同的视角
    ax1.view_init(elev=20, azim=45)
    ax2.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()

def transform_to_robot_frame(global_pose, robot_pose):
    """将全局坐标系下的位姿转换到机器人坐标系下"""
    from scipy.spatial.transform import Rotation
    
    # 提取位置和方向
    global_pos = global_pose[:3]
    global_quat = global_pose[3:]
    robot_pos = robot_pose[:3]
    robot_quat = robot_pose[3:]
    
    # 计算机器人的旋转矩阵
    r_robot = Rotation.from_quat([robot_quat[0], robot_quat[1], robot_quat[2], robot_quat[3]])
    r_robot_mat = r_robot.as_matrix()
    
    # 计算相对位置
    relative_pos = global_pos - robot_pos
    # 将相对位置转换到机器人坐标系
    relative_pos_robot = r_robot_mat.T @ relative_pos
    
    # 计算相对旋转
    r_global = Rotation.from_quat([global_quat[0], global_quat[1], global_quat[2], global_quat[3]])
    relative_rotation = r_robot.inv() * r_global
    relative_quat = relative_rotation.as_quat()
    
    # 组合相对位姿
    relative_pose = np.concatenate([relative_pos_robot, relative_quat])
    return relative_pose

def load_and_print_data(data_root='./collected_data'):
    """
    读取并打印收集的数据
    """
    # 遍历所有数据文件夹
    for episode_dir in sorted(os.listdir(data_root)):
        if not episode_dir.startswith('episode_'):
            continue
            
        print(f"\n{'='*50}")
        print(f"查看数据: {episode_dir}")
        print(f"{'='*50}")
        
        # 读取状态数据
        state_path = os.path.join(data_root, episode_dir, 'state.npz')
        state_data = np.load(state_path, allow_pickle=True)
        state_data = {key: state_data[key].item() if isinstance(state_data[key], np.ndarray) else state_data[key] for key in state_data}

        # 打印指令
        print(f"\n指令: {state_data['instruction']}")
        
        # 收集所有要显示的图像
        images_to_show = []
        print("\n相机观察:")
        for cam_name, obs in state_data['camera_obs'].items():
            print(f"\n{cam_name}相机:")
            if 'rgb_path' in obs and obs['rgb_path']:
                rgb_img = cv2.imread(os.path.join(data_root, episode_dir, obs['rgb_path']))
                print(f"RGB 形状: {rgb_img.shape}")
                images_to_show.append(rgb_img)
            if 'seg_path' in obs and obs['seg_path']:
                seg_img = cv2.imread(os.path.join(data_root, episode_dir, obs['seg_path']), cv2.IMREAD_GRAYSCALE)
                print(f"分割图形状: {seg_img.shape}")
                # 将灰度图转换为3通道图像以便于拼接
                seg_img_color = cv2.cvtColor(seg_img, cv2.COLOR_GRAY2BGR)
                images_to_show.append(seg_img_color)
            if 'points_path' in obs and obs['points_path']:
                points = np.load(os.path.join(data_root, episode_dir, obs['points_path']))
                print(f"点云形状: {points.shape}")

                    
        # 打印机器人状态
        print("\n机器人状态:")
        robot_state = state_data['robot_state']
        for key, value in robot_state.items():
            print(f"{key}: {value}")
            
        # 打印物体状态
        print("\n场景物体状态:")
        for obj_name, obj_state in state_data['object_states'].items():
            print(f"\n{obj_name}:")
            for key, value in obj_state.items():
                print(f"{key}: {value}")
                
        # 读取并打印子目标
        # subgoal 是一个 7 维的向量，表示机器人末端执行器的目标姿态 [x,y,z,qx,qy,qz,qw]
        subgoal_path = os.path.join(data_root, episode_dir, 'subgoal.npy')
        subgoal = np.load(subgoal_path)
        print(f"\n子目标姿态 [x,y,z,qx,qy,qz,qw]:")
        print(subgoal)

        # 获取机器人末端执行器的位姿
        robot_ee_pose = robot_state['ee_pose']
        
        # 转换到机器人坐标系
        relative_subgoal = transform_to_robot_frame(subgoal, robot_ee_pose)
        print(f"\n机器人坐标系下的子目标姿态 [x,y,z,qx,qy,qz,qw]:")
        print(relative_subgoal)
        
        # 可视化两个坐标系下的位姿
        visualize_poses(subgoal, relative_subgoal)
        
        
        # 可以添加cv2.imshow来显示图片
        # 拼接所有图片
        # if images_to_show:
        #     combined_image = np.hstack(images_to_show)  # 水平拼接
        #     cv2.imshow("Combined Image", combined_image)
        #     cv2.waitKey(0)  # 如果要显示图片,取消这行注释
            # cv2.destroyAllWindows()

def set_subgoal_embeddings(llm, dataset):
    for data in dataset:
        embedding = llm.embedding(data['subgoal'])
        data['embedding'] = embedding
    return dataset

if __name__ == "__main__":

    # 数据集生成
    # 阶段1：生成有embedding的数据集，调用llm一次之后保存，以后都用保存的
    # dataset_part_1 = [{
    #     'subgoal': 'reach the red pen and grasp it.',
    #     'grasp_closed': True
    # }, {
    #     'subgoal': 'turn the red pen upright.',
    #     'grasp_closed': True
    # }, {
    #     'subgoal': 'drop the red pen into the black pen holder.',
    #     'grasp_closed': False
    # }]

    # llm = LLM()
    # dataset_with_embedding = set_subgoal_embeddings(llm, dataset_part_1)
    # output_path = os.path.join(ROOT_PATH, '../examples/training/dataset_with_embedding.pkl')
    # pickle.dump(dataset_with_embedding, open(output_path, 'wb'))



    # 阶段2：添加相应图像和夹爪目标位姿
    
    input_path = os.path.join(ROOT_PATH, '../examples/training/dataset_with_embedding.pkl')
    dataset_with_embedding = pickle.load(open(input_path, 'rb'))
    dataset_full = dataset_with_embedding

    data_root = os.path.join(ROOT_PATH, '../Rekep/collected_data')
    data_folder_list = sorted(os.listdir(data_root))
    for data_idx in range(len(dataset_with_embedding)):
        episode_dir = data_folder_list[data_idx]

        # 读取状态数据作为输入
        state_path = os.path.join(data_root, episode_dir, 'state.npz')
        state_data = np.load(state_path, allow_pickle=True)
        state_data = {key: state_data[key].item() if isinstance(state_data[key], np.ndarray) else state_data[key] for key in state_data}

        dataset_full[data_idx]['robot_state'] = state_data['robot_state']

        # 读取相机rgb图像作为输入
        rgb_path = os.path.join(data_root, episode_dir, 'images', '0_rgb.png')
        rgb_img = cv2.imread(rgb_path)
        dataset_full[data_idx]['rgb'] = rgb_img

        # 读取机器人末端位姿作为神经网络的输出label
        subgoal_path = os.path.join(data_root, episode_dir, 'subgoal.npy')
        subgoal = np.load(subgoal_path)
        robot_ee_pose = state_data['robot_state']['ee_pose']
        relative_subgoal = transform_to_robot_frame(subgoal, robot_ee_pose)
        dataset_full[data_idx]['label'] = relative_subgoal

    # 保存数据集
    output_path = os.path.join(ROOT_PATH, '../examples/training/dataset_full.pkl')
    pickle.dump(dataset_full, open(output_path, 'wb'))

    print(dataset_full)