import grpc
import btgym.proto.simulator_pb2 as simulator_pb2
import btgym.proto.simulator_pb2_grpc as simulator_pb2_grpc

class SimulatorClient:
    def __init__(self):
        self.channel = grpc.insecure_channel('localhost:50051')
        self.stub = simulator_pb2_grpc.SimulatorServiceStub(self.channel)

    def load_task(self, task_name):
        request = simulator_pb2.LoadTaskRequest(task_name=task_name)
        response = self.stub.LoadTask(request)
        return response.success, response.message

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

def main():
    client = SimulatorClient()
    
    # 测试加载任务
    success, msg = client.load_task('turning_out_all_lights_before_sleep')
    print(f"加载任务结果: {success}, {msg}")

    # 初始化动作原语
    success, msg = client.init_action_primitives()
    print(f"初始化动作原语结果: {success}, {msg}")

    # 获取场景名称
    scene_name = client.get_scene_name()
    print(f"场景名称: {scene_name}")

    # 获取机器人位置
    robot_pos = client.get_robot_pos()
    print(f"机器人位置: {robot_pos}")

    # 导航到物体
    success, msg = client.navigate_to_object("light_switch")
    print(f"导航结果: {success}, {msg}")

if __name__ == '__main__':
    main() 