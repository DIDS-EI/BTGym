import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from btgym import cfg
import pickle
import os
from btgym import ROOT_PATH
import torch as th


# RGB图像，目标物体，目标物体的
# 输出： 7维的向量，表示7个关节的角度



class RobotDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        # 处理输入数据
        embedding = torch.tensor(data['embedding'], dtype=torch.float32)
        joint_positions = data['robot_state']['joint_positions']
        ee_pose = torch.tensor(data['robot_state']['ee_pose'], dtype=torch.float32)
        gripper_state = torch.tensor([float(data['robot_state']['gripper_state'])], dtype=torch.float32)
        
        # 处理RGB图像
        rgb = torch.tensor(data['rgb'], dtype=torch.float32).permute(2, 0, 1) / 255.0  # 归一化并转换为CxHxW格式
        
        # 组合机器人状态
        robot_state = torch.cat([joint_positions, ee_pose, gripper_state])
        
        # 处理标签
        label = torch.tensor(data['label'], dtype=torch.float32)
        
        return {
            'embedding': embedding,
            'robot_state': robot_state,
            'rgb': rgb,
            'label': label
        }

class RobotControlNet(nn.Module):
    def __init__(self):
        super(RobotControlNet, self).__init__()
        
        # 视觉语言嵌入处理网络
        self.embedding_net = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # RGB图像处理网络 - 修改CNN架构
        self.cnn = nn.Sequential(
            # 输入: 3x480x480
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),  # -> 32x120x120
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 32x60x60
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> 64x30x30
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 64x15x15
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # -> 128x15x15
            nn.ReLU(),
            nn.MaxPool2d(3),  # -> 128x5x5
            
            nn.Flatten(),  # -> 128*5*5
            nn.Linear(128*5*5, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # 机器人状态处理网络
        self.robot_state_net = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256)
        )
        
        # 特征融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(768, 512),  # 256(embedding) + 256(rgb) + 256(robot_state)
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )
        
    def forward(self, embedding, robot_state, rgb):
        embedding_features = self.embedding_net(embedding)
        rgb_features = self.cnn(rgb)
        robot_features = self.robot_state_net(robot_state)
        
        # 组合所有特征
        combined_features = torch.cat([embedding_features, rgb_features, robot_features], dim=1)
        output = self.fusion_net(combined_features)
        
        return output

def train(dataset, output_path, batch_size=32, num_epochs=1000, learning_rate=1e-4):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据加载器
    train_dataset = RobotDataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始��模型
    model = RobotControlNet().to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            # 获取数据
            embedding = batch['embedding'].to(device)
            robot_state = batch['robot_state'].to(device)
            rgb = batch['rgb'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(embedding, robot_state, rgb)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 打印训练进度
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    # 保存模型
    torch.save(model.state_dict(), output_path)
    print("训练完成，模型已保存。")

if __name__ == "__main__":
    dataset = pickle.load(open(os.path.join(ROOT_PATH, '../examples/training/dataset_full.pkl'), 'rb'))
    """
    for data in dataset:
        data['embedding'].shape = (1, 1572)
        data['robot_state'].shape = (1, 10)
        data['rgb'].shape = (480, 480, 3)
        data['label'].shape = (1, 7)
    """
    assert isinstance(dataset[0]['embedding'], list) and len(dataset[0]['embedding']) == 1536
    assert isinstance(dataset[0]['robot_state'], dict)
    assert isinstance(dataset[0]['robot_state']['joint_positions'], th.Tensor) and dataset[0]['robot_state']['joint_positions'].shape == (8,)
    assert isinstance(dataset[0]['robot_state']['ee_pose'], np.ndarray) and dataset[0]['robot_state']['ee_pose'].shape == (7,)
    assert isinstance(dataset[0]['robot_state']['gripper_state'], bool)
    assert isinstance(dataset[0]['rgb'], np.ndarray) and dataset[0]['rgb'].shape == (480, 480, 3)
    assert isinstance(dataset[0]['label'], np.ndarray) and dataset[0]['label'].shape == (7,)

    output_path = os.path.join(ROOT_PATH, '../examples/training/subgoal_pose_net.pth')
    train(dataset, output_path)