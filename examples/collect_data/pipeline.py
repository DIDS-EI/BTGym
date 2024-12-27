
"""
# 1. 在其他进程中启动simulator server
# ```shell
# python btgym/simulator/launch_simulator_server.py
# ```
"""

"""
# 2. 在其他进程中启动molmo server
# ```shell
# python btgym/molmo/launch_molmo_server.py
# ```
"""

"""
# 3. 调用LLM生成任务

from btgym.llm.llm import LLM
from btgym.llm.generate_bddl import generate_bddl

llm = LLM()

# 如果想随机选择场景，可以取消注释以下代码
# import random
# with open(f"{cfg.ASSETS_PATH}/scene_list.txt", "r") as f:
#     scene_list = [line.strip() for line in f.readlines()]
# scene_name = random.choice(scene_list)

scene_name = "Rs_int"

bddl = generate_bddl(llm, scene_name)
# 按照命名规则保存bddl （暂未实现）
"""



"""
# 4. 在仿真器中采样任务


from btgym.simulator.simulator_client import SimulatorClient
simulator_client = SimulatorClient()

json_path = ''
while json_path == '':
    json_path = simulator_client.call(func='SampleTask', task_name='test_task')
"""


"""
# 5. 在仿真器中读取任务

from btgym.simulator.simulator_client import SimulatorClient
simulator_client = SimulatorClient()

simulator_client.call(func='LoadCustomTask', 
task_name='test_task',json_path=json_path)

"""


"""
# 6. 开始执行任务

"""


"""
# xxx. 保存成功数据
"""