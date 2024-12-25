import grpc
from concurrent import futures
import multiprocessing as mp
from queue import Empty
from typing import Any, Dict, Type

import btgym.simulator.simulator_pb2 as simulator_pb2
import btgym.simulator.simulator_pb2_grpc as simulator_pb2_grpc
from btgym.simulator.simulator import Simulator
import numpy as np

class RPCMethod:
    """RPC方法注册器"""
    registry = {}

    def __init__(self, response_type: Type):
        self.response_type = response_type

    def __call__(self, func):
        method_name = func.__name__
        RPCMethod.registry[method_name] = {
            'handler': func,
            'response_type': self.response_type
        }

        def servicer_method(servicer, request, context):
            try:
                servicer._track_connection(context)
                servicer.command_queue.put((method_name, request))
                result = servicer.result_queue.get(timeout=10)
                return self.response_type(**result) if result else self.response_type()
            except (Empty, Exception):
                return self.response_type()

        RPCMethod.registry[method_name]['servicer_method'] = servicer_method
        return func

class SimulatorServicer(simulator_pb2_grpc.SimulatorServiceServicer):
    def __init__(self, command_queue: mp.Queue, result_queue: mp.Queue):
        self.command_queue = command_queue
        self.result_queue = result_queue
        self._active_connections = set()
        
    def _track_connection(self, context):
        peer = context.peer()
        self._active_connections.add(peer)
        context.add_callback(lambda: self._handle_disconnect(peer))
        return peer
    
    def _handle_disconnect(self, peer):
        self._active_connections.remove(peer)
        self.command_queue.put(('client_disconnected', peer))

class SimulatorCommandHandler:
    def __init__(self, simulator: Simulator):
        self.simulator = simulator

    def handle_command(self, command: str, request: Any) -> Dict:
        if command not in RPCMethod.registry:
            return {}
        
        try:
            handler = RPCMethod.registry[command]['handler']
            return handler(self, request) or {}
        except Exception as e:
            print(f"Error handling command {command}: {str(e)}")
            return {}

    @RPCMethod(simulator_pb2.Empty)
    def LoadTask(self, request):
        self.simulator.load_task(request.task_name)

    @RPCMethod(simulator_pb2.NavigateToObjectRequest)
    def NavigateToObject(self, request):
        self.simulator.navigate_to_object(request.object_name)

    @RPCMethod(simulator_pb2.Empty)
    def InitActionPrimitives(self, request):
        self.simulator.init_action_primitives()

    @RPCMethod(simulator_pb2.SceneNameResponse)
    def GetSceneName(self, request) -> Dict:
        return {'scene_name': self.simulator.get_scene_name()}

    @RPCMethod(simulator_pb2.RobotPosResponse)
    def GetRobotPos(self, request) -> Dict:
        return {'position': self.simulator.get_robot_pos()}

    @RPCMethod(simulator_pb2.GetRobotJointStatesResponse)
    def GetRobotJointStates(self, request) -> Dict:
        joint_positions = self.simulator.get_joint_states()
        return {'joint_states': joint_positions.tolist() if isinstance(joint_positions, np.ndarray) else joint_positions}

    @RPCMethod(simulator_pb2.SetRobotJointStatesRequest)
    def SetRobotJointStates(self, request) -> Dict:
        self.simulator.set_joint_states(request.joint_states)


    @RPCMethod(simulator_pb2.EEFPoseResponse)
    def GetRobotEEFPose(self, request) -> Dict:
        pose = self.simulator.get_end_effector_pose()
        return {'eef_pose': pose.tolist() if isinstance(pose, np.ndarray) else pose}

    @RPCMethod(simulator_pb2.RelativeEEFPoseResponse)
    def GetRelativeEEFPose(self, request) -> Dict:
        pose = self.simulator.get_relative_eef_pose()
        return {'relative_eef_pose': pose.tolist() if isinstance(pose, np.ndarray) else pose}

    @RPCMethod(simulator_pb2.TaskObjectsResponse)
    def GetTaskObjects(self, request) -> Dict:
        objects = self.simulator.get_task_objects()
        return {'object_names': objects}

    @RPCMethod(simulator_pb2.Empty)
    def GraspObject(self, request) -> Dict:
        self.simulator.grasp_object(request.object_name)

    @RPCMethod(simulator_pb2.ReachPoseRequest)
    def ReachPose(self, request) -> Dict:
        self.simulator.reach_pose(request.pose, request.is_local)

    @RPCMethod(simulator_pb2.SaveCameraImageRequest)
    def SaveCameraImage(self, request) -> Dict:
        self.simulator.save_camera_image(request.output_path)

    @RPCMethod(simulator_pb2.SetTargetVisualPoseRequest)
    def SetTargetVisualPose(self, request) -> Dict:
        self.simulator.set_target_visual_pose(request.pose)

    @RPCMethod(simulator_pb2.GetCameraInfoResponse)
    def GetCameraInfo(self, request) -> Dict:
        camera_info = self.simulator.get_camera_info()
        return camera_info

    @RPCMethod(simulator_pb2.GetObsResponse)
    def GetObs(self, request) -> Dict:
        obs = self.simulator.get_obs()
        return obs


# 动态添加方法到ServicerClass
for method_name, method_info in RPCMethod.registry.items():
    setattr(SimulatorServicer, method_name, method_info['servicer_method'])

def main_process_loop(command_queue: mp.Queue, result_queue: mp.Queue):
    simulator = Simulator()
    handler = SimulatorCommandHandler(simulator)
    
    while True:
        try:
            if command_queue.empty():
                simulator.idle_step()
                continue
                
            command, request = command_queue.get()
            if command == 'client_disconnected':
                print(f"客户端断开连接: {request}")
                continue
                
            result = handler.handle_command(command, request)
            result_queue.put(result)
        except Exception as e:
            print(f"Error handling command {command}: {str(e)}")

def serve():
    command_queue = mp.Queue()
    result_queue = mp.Queue()
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = SimulatorServicer(command_queue, result_queue)
    simulator_pb2_grpc.add_SimulatorServiceServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:50052')
    server.start()
    
    try:
        main_process_loop(command_queue, result_queue)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    serve()