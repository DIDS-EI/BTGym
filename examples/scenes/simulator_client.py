import grpc
import btgym.simulator.simulator_pb2 as simulator_pb2
import btgym.simulator.simulator_pb2_grpc as simulator_pb2_grpc
import numpy as np

class SimulatorClient:
    def __init__(self, timeout=10):
        self.channel = grpc.insecure_channel('localhost:50051')
        self.stub = simulator_pb2_grpc.SimulatorServiceStub(self.channel)
        self.timeout = timeout  # 添加超时设置

    def load_task(self, task_name):
        try:
            request = simulator_pb2.LoadTaskRequest(task_name=task_name)
            response = self.stub.LoadTask(
                request, 
                timeout=self.timeout
            )
            return response.success, response.message
        except grpc.RpcError as e:
            return False, f"RPC错误: {str(e)}"

    def init_action_primitives(self):
        response = self.stub.InitActionPrimitives(simulator_pb2.Empty())
        return response.success, response.message

    def navigate_to_object(self, object_name):
        request = simulator_pb2.NavigateRequest(object_name=object_name)
        response = self.stub.NavigateToObject(request)
        return response.success, response.message

    def get_scene_name(self):
        response = self.stub.GetSceneName(simulator_pb2.Empty())
        return response.scene_name

    def get_robot_pos(self):
        response = self.stub.GetRobotPos(simulator_pb2.Empty())
        return response.position

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
    success, msg = client.load_task('putting_shoes_on_rack')
    print(f"加载任务结果: {success}, {msg}")

    # navigate_to_object_result = client.navigate_to_object("light_switch")
    # print(f"导航结果: {navigate_to_object_result}")
    # 初始化动作原语
    # success, msg = client.init_action_primitives()
    # print(f"初始化动作原语结果: {success}, {msg}")

    # # 获取场景名称
    # scene_name = client.get_scene_name()
    # print(f"场景名称: {scene_name}")

    # # 获取机器人位置
    robot_pos = client.get_robot_pos()
    print(f"机器人位置: {robot_pos}")

    # # 导航到物体
    # success, msg = client.navigate_to_object("light_switch")
    # print(f"导航结果: {success}, {msg}")

    # 测试获取图像
    rgb, depth = client.get_rgbd()
    if rgb is not None:
        print(f"获取到RGB图像，形状: {rgb.shape}")
        print(f"获取到深度图像，形状: {depth.shape}")

if __name__ == '__main__':
    main() 