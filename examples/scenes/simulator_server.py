import grpc
from concurrent import futures
import time

import btgym.proto.simulator_pb2 as simulator_pb2
import btgym.proto.simulator_pb2_grpc as simulator_pb2_grpc
from view_scene import Simulator

class SimulatorServicer(simulator_pb2_grpc.SimulatorServiceServicer):
    def __init__(self):
        self.simulator = Simulator()

    def LoadTask(self, request, context):
        try:
            self.simulator.load_behavior_task_by_name(request.task_name)
            return simulator_pb2.CommonResponse(success=True, message="任务加载成功")
        except Exception as e:
            return simulator_pb2.CommonResponse(success=False, message=str(e))

    def InitActionPrimitives(self, request, context):
        try:
            self.simulator.init_action_primitives()
            return simulator_pb2.CommonResponse(success=True, message="动作原语初始化成功")
        except Exception as e:
            return simulator_pb2.CommonResponse(success=False, message=str(e))

    def NavigateToObject(self, request, context):
        try:
            self.simulator.navigate_to_object(request.object_name)
            return simulator_pb2.CommonResponse(success=True, message="导航指令已发送")
        except Exception as e:
            return simulator_pb2.CommonResponse(success=False, message=str(e))

    def GetSceneName(self, request, context):
        try:
            scene_name = self.simulator.get_scene_name()
            return simulator_pb2.SceneNameResponse(scene_name=scene_name)
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def GetRobotPos(self, request, context):
        try:
            pos = self.simulator.get_robot_pos()
            return simulator_pb2.RobotPosResponse(position=pos.tolist())
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def Step(self, request, context):
        try:
            self.simulator.step()
            return simulator_pb2.CommonResponse(success=True, message="执行成功")
        except Exception as e:
            return simulator_pb2.CommonResponse(success=False, message=str(e))

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    simulator_pb2_grpc.add_SimulatorServiceServicer_to_server(
        SimulatorServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("服务器启动在端口50051...")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve() 