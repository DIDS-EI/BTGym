import grpc
import sys
from btgym import cfg
sys.path.append(f'{cfg.ROOT_PATH}/simulator')
import btgym.simulator.simulator_pb2 as simulator_pb2
import btgym.simulator.simulator_pb2_grpc as simulator_pb2_grpc
import numpy as np
import json
from btgym.utils import og_utils
import torch as th


class SimulatorClient:
    def __init__(self):
        i = 1

        self.camera_info = None
        self.obs = None
        while True: 
            try:
                self.channel = grpc.insecure_channel('192.168.2.78:50052')
                # self.channel = grpc.insecure_channel('localhost:50052')
                # 设置5秒超时
                grpc.channel_ready_future(self.channel).result(timeout=5)
                self.stub = simulator_pb2_grpc.SimulatorServiceStub(self.channel)
                print(f"连接仿真器成功！")
                break
            except grpc.FutureTimeoutError:
                print(f"连接仿真器超时，重试第{i}次")
            except Exception as e:
                print(f"连接仿真器错误，错误信息: {str(e)}，重试第{i}次")
            i += 1

    def call(self, func,**kwargs):
        try:
            if kwargs:
                request = getattr(simulator_pb2, func+"Request")(**kwargs)
            else:
                request = simulator_pb2.Empty()
            response = getattr(self.stub, func)(request)
            return response
        except Exception as e:
            print(f"调用仿真器函数失败，错误信息: {str(e)}")
            return None

    def load_task(self, task_name):
        request = simulator_pb2.LoadTaskRequest(task_name=task_name)
        response = self.stub.LoadTask(request)

    def init_action_primitives(self):
        response = self.stub.InitActionPrimitives(simulator_pb2.Empty())

    def navigate_to_object(self, object_name):
        request = simulator_pb2.NavigateRequest(object_name=object_name)
        response = self.stub.NavigateToObject(request)

    def get_scene_name(self):
        response = self.stub.GetSceneName(simulator_pb2.Empty())
        return response.scene_name

    def get_robot_pos(self):
        response = self.stub.GetRobotPos(simulator_pb2.Empty())
        return response.position

    def get_robot_joint_states(self):
        response = self.stub.GetRobotJointStates(simulator_pb2.Empty())
        return response.joint_states

    def get_robot_eef_pose(self):
        response = self.stub.GetRobotEEFPose(simulator_pb2.Empty())
        return response.eef_pose

    def step(self):
        response = self.stub.Step(simulator_pb2.Empty())
        return response.success, response.message

    def get_rgbd(self):
        """获取RGBD图像
        
        Returns:
            tuple: (rgb_array, depth_array) - RGB和深度图像的numpy数组
        """
        try:
            response = self.stub.GetRGBD(simulator_pb2.Empty())
            
            # 将bytes转回numpy数组
            rgb = np.frombuffer(response.rgb, dtype=np.uint8).reshape(
                response.height, response.width, response.channels
            )
            depth = np.frombuffer(response.depth, dtype=np.float32).reshape(
                response.height, response.width
            )
            
            return rgb, depth
        except grpc.RpcError as e:
            print(f"获取图像失败: {str(e)}")
            return None, None

    def get_obs(self):
        raw_obs= self.call(func='GetObs')

        rgb = np.frombuffer(raw_obs.rgb, dtype=np.uint8)
        rgb = rgb.reshape((480, 480, 4))

        depth = np.frombuffer(raw_obs.depth, dtype=np.float32)
        depth = depth.reshape((480, 480))

        seg_semantic = np.frombuffer(raw_obs.seg_semantic, dtype=np.int32)
        seg_semantic = seg_semantic.reshape((480, 480))

        self.obs = {
            'rgb': rgb,
            'depth': depth,
            'seg_semantic': seg_semantic,
            'seg_info': json.loads(raw_obs.seg_info),
            'proprio': np.array(raw_obs.proprio)
        }

    def get_camera_info(self):
        camera_info = self.call(func='GetCameraInfo')
        intrinsics = np.array(camera_info.intrinsics).reshape(3,3)
        extrinsics = th.tensor(camera_info.extrinsics).reshape(4,4)

        self.camera_info = {
            'intrinsics': intrinsics,
            'extrinsics': extrinsics
        }

    def pixel_to_world(self, pixel_x, pixel_y):
        p2w = og_utils.pixel_to_3d_points(self.obs['depth'], self.camera_info['intrinsics'], self.camera_info['extrinsics'])
        return p2w[pixel_y,pixel_x]

    def pixel_to_world1(self, pixel_x, pixel_y):
        """
        将图像上的像素点转换为世界坐标系中的3D点
        
        Args:
            pixel_x (int): 像素x坐标
            pixel_y (int): 像素y坐标  
        
        Returns:
            np.array: 3D点在世界坐标系中的坐标 [x, y, z]
        """
        # 1. 获取图像分辨率
        image_height = image_width = self.camera_info.renderProductResolution

        # 2. 获取该像素的深度值
        depth = self.obs['depth'][pixel_y, pixel_x]
        print('depth', depth)
        # 3. 计算归一化设备坐标(NDC)
        # 注意y坐标需要翻转，因为图像坐标系原点在左上角
        ndc_x = (2.0 * pixel_x / image_width - 1.0)
        ndc_y = -(2.0 * pixel_y / image_height - 1.0)  # 翻转y轴
        ndc_z = 2.0 * depth - 1.0  # 将深度值转换到NDC空间
        
        # 4. 构建NDC空间中的点
        ndc_point = np.array([ndc_x, ndc_y, ndc_z, 1.0])
        print('ndc_point', ndc_point)

        # 5. 获取投影矩阵和视图矩阵
        proj_matrix = np.array(self.camera_info.cameraProjection).reshape(4, 4)
        view_matrix = np.array(self.camera_info.cameraViewTransform).reshape(4, 4)
        
        print('proj_matrix', proj_matrix)
        print('view_matrix', view_matrix)

        # 6. 计算投影-视图矩阵的逆矩阵
        vp_matrix = view_matrix @ proj_matrix
        vp_inv = np.linalg.inv(vp_matrix)
        
        # 7. 将NDC点转换到世界坐标系
        world_point = vp_inv @ ndc_point
        world_point = world_point / world_point[3]  # 透视除法
        
        return world_point[:3]  # 返回xyz坐标


def main():
    client = SimulatorClient()
    
    # 测试加载任务
    response = client.call(func='LoadTask', task_name='putting_shoes_on_rack')
    # response = client.call(func='NavigateToObject', object_name='shelf.n.01_1')

    client.get_obs()
    client.get_camera_info()
    pos = client.pixel_to_world(40,140)
    # # print(pos)
    pos = [0,1,1]
    response = client.call(func='SetTargetVisualPose', pose=[*pos, 0, 0, 0])
    response = client.call(func='SetCameraLookatPos', pos=pos)


    # 关节状态 limits
    # tensor([-1.0000e+03, -1.0000e+03,  0.0000e+00, -1.5700e+00, -1.6056e+00,
    #     -7.6000e-01, -1.2210e+00, -6.2832e+00, -2.2510e+00, -6.2832e+00,
    #     -2.1600e+00, -6.2832e+00,  0.0000e+00,  0.0000e+00])

    # tensor([1.0000e+03, 1.0000e+03, 3.8615e-01, 1.5700e+00, 1.6056e+00, 1.4500e+00,
    #         1.5180e+00, 6.2832e+00, 2.2510e+00, 6.2832e+00, 2.1600e+00, 6.2832e+00,
    #         5.0000e-02, 5.0000e-02])

    # response = client.call(func='GetTaskObjects')
    # print(response)
    # response = client.call(func='SetRobotJointStates',
    #                        joint_states=[
    #             0.0,
    #             0.0,  # wheels
    #             0.0,  # trunk
    #             1.2,
    #             0,
    #             -2,  # head
    #             -0.8,
    #             1.7,
    #             2.0,
    #             -1.0,
    #             1.36904,
    #             1.90996,  # arm
    #             0.05,
    #             0.05,  # gripper
    #         ])
    # print(response)
    # response = client.call(func='ReachPose',pose=[0.6,-0.2,1.418,180,0,0],is_local=True)
    # print(response)
    # response = client.call(func='GraspObject', object_name='gym_shoe.n.01_1')
    
    # # ��试获取图像
    # rgb, depth = client.get_rgbd()
    # if rgb is not None:
    #     print(f"获取到RGB图像，形状: {rgb.shape}")
    #     print(f"获取到深度图像，形状: {depth.shape}")

if __name__ == '__main__':
    main() 