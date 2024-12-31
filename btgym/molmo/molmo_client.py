import grpc
import sys
from btgym import cfg
sys.path.append(f'{cfg.ROOT_PATH}/molmo')
import btgym.molmo.molmo_pb2 as molmo_pb2
import btgym.molmo.molmo_pb2_grpc as molmo_pb2_grpc
import numpy as np
import os
import re

class MolmoClient:
    def __init__(self):
        i = 1
        while True: 
            try:
                self.channel = grpc.insecure_channel('localhost:50053')
                # 设置5秒超时
                grpc.channel_ready_future(self.channel).result(timeout=5)
                self.stub = molmo_pb2_grpc.MolmoServiceStub(self.channel)
                print(f"连接Molmo成功！")
                break
            except grpc.FutureTimeoutError:
                print(f"连接Molmo超时，重试第{i}次")
            except Exception as e:
                print(f"连接Molmo错误，错误信息: {str(e)}，重试第{i}次")
            i += 1

    def call(self, func,**kwargs):
        try:
            if kwargs:
                request = getattr(molmo_pb2, func+"Request")(**kwargs)
            else:
                request = molmo_pb2.Empty()
            response = getattr(self.stub, func)(request)
            return response
        except Exception as e:
            print(f"调用Molmo函数失败，错误信息: {str(e)}")
            return None

    def extract_points(self, molmo_output, image):
        all_points = []
        for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', molmo_output):
            try:
                point = [float(match.group(i)) for i in range(1, 3)]
            except ValueError:
                pass
            else:
                point = np.array(point)
                if np.max(point) > 100:
                    # Treat as an invalid output
                    continue
                point /= 100.0
                point = point * np.array([image.width, image.height])
                all_points.append(point)
        return all_points

def main():
    client = MolmoClient()
    DIR = os.path.dirname(os.path.abspath(__file__))
    # 测试加载任务
    # response = client.call(func='LoadTask', task_name='putting_shoes_on_rack')
    # response = client.call(func='GetTaskObjects')
    # print(response)
    response = client.call(func='PointQA',
                           query='reorient the white pen and drop it upright into the black pen holder.',
                           image_path=f'{DIR}/camera_0_rgb.png',
                        )
    print(response)

if __name__ == '__main__':
    main() 