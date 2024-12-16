import grpc
from concurrent import futures
import time
import multiprocessing as mp
from queue import Empty
import functools
from typing import Any, Callable, Dict, Tuple, Type

import btgym.simulator.simulator_pb2 as simulator_pb2
import btgym.simulator.simulator_pb2_grpc as simulator_pb2_grpc
from btgym.simulator.simulator import Simulator
import numpy as np

class RPCMethod:
    """RPC方法注册器"""
    registry = {}

    def __init__(self, response_type: Type, simulator_method: str = None):
        """
        Args:
            response_type: 返回消息的类型
            simulator_method: 模拟器中对应的方法名，如果为None则使用处理函数名
        """
        self.response_type = response_type
        self.simulator_method = simulator_method

    def __call__(self, func):
        method_name = func.__name__
        
        # 注册处理函数
        RPCMethod.registry[method_name] = {
            'handler': func,
            'response_type': self.response_type,
            'simulator_method': self.simulator_method or method_name
        }

        # 创建Servicer方法
        def servicer_method(servicer, request, context):
            try:
                servicer.command_queue.put((method_name, request))
                try:
                    result = servicer.result_queue.get(timeout=10)
                    if isinstance(self.response_type(), simulator_pb2.CommonResponse):
                        return self.response_type(
                            success=result['success'], 
                            message=result.get('message', '')
                        )
                    return self.response_type(**result)
                except Empty:
                    return self.response_type(success=False, message="操作超时")
            except Exception as e:
                if isinstance(self.response_type(), simulator_pb2.CommonResponse):
                    return self.response_type(success=False, message=str(e))
                return self.response_type()

        # 将原始处理函数和Servicer方法都保存到注册表中
        RPCMethod.registry[method_name]['servicer_method'] = servicer_method
        
        # 返回原始处理函数
        return func

class SimulatorServicer(simulator_pb2_grpc.SimulatorServiceServicer):
    """自动生成的服务类"""
    def __init__(self, command_queue: mp.Queue, result_queue: mp.Queue):
        self.command_queue = command_queue
        self.result_queue = result_queue

class SimulatorCommandHandler:
    """处理模拟器命令的类"""
    def __init__(self, simulator: Simulator):
        self.simulator = simulator

    def handle_command(self, command: str, request: Any) -> Dict:
        """处理命令的统一入口"""
        if command not in RPCMethod.registry:
            return {'success': False, 'message': f'未知命令: {command}'}
        
        method_info = RPCMethod.registry[command]
        handler = method_info['handler']
        try:
            result = handler(self, request)
            if isinstance(result, dict):
                return result
            # 如果返回的是protobuf消息，转换为字典
            return {
                'success': getattr(result, 'success', True),
                'message': getattr(result, 'message', '')
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # 在这里使用装饰器定义所有RPC方法
    @RPCMethod(simulator_pb2.CommonResponse, 'load_behavior_task_by_name')
    def LoadTask(self, request) -> Dict:
        """加载任务"""
        self.simulator.load_behavior_task_by_name(request.task_name)
        return {'success': True}

    @RPCMethod(simulator_pb2.CommonResponse)
    def InitActionPrimitives(self, request) -> Dict:
        """初始化动作原语"""
        # 实现初始化动作原语的逻辑
        return {'success': True}

    @RPCMethod(simulator_pb2.SceneNameResponse)
    def GetSceneName(self, request) -> Dict:
        """获取场景名称"""
        scene_name = self.simulator.get_scene_name()
        return {'scene_name': scene_name}

    @RPCMethod(simulator_pb2.RobotPosResponse)
    def GetRobotPos(self, request) -> Dict:
        """获��机器人位置"""
        pos = self.simulator.get_robot_pos()
        return {'position': pos}

    @RPCMethod(simulator_pb2.ImageResponse)
    def GetRGBD(self, request) -> Dict:
        """获取RGBD图像"""
        rgb, depth = self.simulator.get_camera_images()  # 假设simulator有这个方法
        
        # 将numpy数组转换为bytes
        rgb_bytes = rgb.tobytes()
        depth_bytes = depth.tobytes()
        
        return {
            'rgb': rgb_bytes,
            'depth': depth_bytes,
            'height': rgb.shape[0],
            'width': rgb.shape[1],
            'channels': rgb.shape[2] if len(rgb.shape) > 2 else 1
        }

# 动态添加方法到ServicerClass
for method_name, method_info in RPCMethod.registry.items():
    setattr(SimulatorServicer, method_name, method_info['servicer_method'])

def main_process_loop(command_queue: mp.Queue, result_queue: mp.Queue):
    """主进程循环处理命令"""
    simulator = Simulator()
    handler = SimulatorCommandHandler(simulator)
    
    while True:
        try:
            if command_queue.empty():
                simulator.idle_step()
                continue
            command, request = command_queue.get()
            result = handler.handle_command(command, request)
            result_queue.put(result)
        except Exception as e:
            print(f"主进程错误: {e}")
            result_queue.put({'success': False, 'error': str(e)})

def run_grpc_server(command_queue: mp.Queue, result_queue: mp.Queue):
    """在子进程中运行gRPC服务器"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = SimulatorServicer(command_queue, result_queue)
    simulator_pb2_grpc.add_SimulatorServiceServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("服务器启动在端口50051...")
    server.wait_for_termination()

def serve():
    command_queue = mp.Queue()
    result_queue = mp.Queue()
    
    server_process = mp.Process(
        target=run_grpc_server, 
        args=(command_queue, result_queue)
    )
    server_process.daemon = True
    server_process.start()

    try:
        main_process_loop(command_queue, result_queue)
    except KeyboardInterrupt:
        print("正在关闭服务器...")
        server_process.terminate()
        server_process.join()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    serve()