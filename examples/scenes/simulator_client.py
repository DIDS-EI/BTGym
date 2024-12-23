import grpc
import sys
from btgym import cfg
sys.path.append(f'{cfg.ROOT_PATH}/simulator')
import btgym.simulator.simulator_pb2 as simulator_pb2
import btgym.simulator.simulator_pb2_grpc as simulator_pb2_grpc
import numpy as np

class SimulatorClient:
    def __init__(self):
        i = 1
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

def main():
    client = SimulatorClient()
    
    # 测试加载任务
    response = client.call(func='LoadTask', task_name='test_task')
    response = client.call(func='NavigateToObject', object_name='gym_shoe.n.01_1')





    # response = client.call(func='GetTaskObjects')
    # print(response)
    # response = client.call(func='SetRobotJointStates',
    #                        joint_states=[
    #             0.0,
    #             0.0,  # wheels
    #             0.0,  # trunk
    #             0.0,
    #             -1.5,
    #             0.0,  # head
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
    
    # # 测试获取图像
    # rgb, depth = client.get_rgbd()
    # if rgb is not None:
    #     print(f"获取到RGB图像，形状: {rgb.shape}")
    #     print(f"获取到深度图像，形状: {depth.shape}")

if __name__ == '__main__':
    main() 