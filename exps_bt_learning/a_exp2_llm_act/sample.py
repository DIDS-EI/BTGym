from btgym.simulator.simulator import Simulator
import pathlib
DIR = pathlib.Path(__file__).parent
import os
import shutil


act_str_ls = ["Walk(apple)","Grasp(apple)","Walk(coffee_table)","Place(apple,coffee_table)"]
act_str = "Walk(apple)"





#########################
# 1. 输入是一个动作:谓词+物体
#########################
# 1.1 导入环境 config

task_name = "task3"
scene_file_name = "scene_file_0"
folder_path  = os.path.join(DIR.parent, "tasks")
json_path = os.path.join(DIR.parent, "tasks", task_name, scene_file_name)

simulator = Simulator(headless=False)
# simulator.load_custom_task(folder_path= os.path.join(DIR.parent, "tasks"), task_name=task_name, scene_file_name=scene_file_name)
# simulator.load_from_json(task_name=task_name, json_path=json_path)







# 直接load不采样和保存的
# 采样准备:
# 1. 一个 BDDL 文件,可考虑从已有的BDDL中改写
# 2. 场景 scene_name ,来自于 50个场景(要支持的场景(查看下面链接))
# 所有任务：
# https://behavior.stanford.edu/knowledgebase/tasks/
# output_json_path = simulator.load_custom_task(task_name=task_name, scene_name="house_single_floor",scene_file_name=scene_file_name,\
#                                                 folder_path=os.path.join(DIR.parent, f"tasks"),is_sample=True)
# print("output_json_path:",output_json_path)


# 加载
# 采样完成后直接加载
simulator.load_custom_task(task_name=task_name, scene_name="Beechwood_0_garden",scene_file_name=scene_file_name,\
                                                folder_path=os.path.join(DIR.parent, f"tasks"),is_sample=False)
# target_object_type = "fridge"
# 获取在环境中的真实名字
all_objects = simulator.get_available_objects()
print("all_objects:",all_objects)

simulator.navigate_to_object(object_name=f"printer.n.03_1")

# for i in range(3,8):
#     try:
#         simulator.navigate_to_object(object_name=f"switch.n.01_{i}")
#         simulator.idle_step(5)
#         print("i:",i)
#     except Exception as e:
#         print(e)
#         continue
simulator.idle()















#######################################################################


# 采样
# 输入: 任务名, 场景名, 输出json路径
# 1. 准备一个 BDDL 文件,可考虑从已有的BDDL中改写
# 2. 选择一个场景,来自于 50个场景
# Rs_garden 有 electric_switch_wseglt_2
# house_single_floor 有冰箱
# school_computer_lab_and_infirmary 有开关 turning_out_all_lights_before_sleep

# sample_custom_task
# 1. 选择已有的 bddl 和支持的场景(查看下面链接)
# 2. 运行采样得到 json 保存到 bddl 的文件下,加载

# 所有任务：
# https://behavior.stanford.edu/knowledgebase/tasks/

# task_name="task3"
# output_json_path = simulator.sample_custom_task(task_name=task_name, scene_name="house_single_floor",\
#                                                 folder_path = folder_path,output_json_path=json_path)

# output_json_path = simulator.sample_custom_task(task_name="switch", scene_name="house_single_floor")
# # 将生成的文件拷贝到 DIR/../btgym/assets/my_tasks/{task_name}/problem0.bddl 下作为 scene_file_0.json
# # 复制文件并重命名为scene_file_0.json
# target_path = os.path.join(DIR.parent, f"tasks/{task_name}", f'{scene_file_name}.json')
# shutil.copy2(json_path, target_path)
# print('已复制场景文件到:', target_path)
# # print("output_json_path:",output_json_path)
















#########################
# 2. 执行动作
#########################
# walk pick placeOn palceIn open close switchOn switchOff
# 使用工具: 
simulator.idle()






# 想 5 个场景
# pick & place
# open close