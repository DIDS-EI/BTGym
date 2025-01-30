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

task_name = "aaa_demo0_draw6" #"aaa_demo0_draw3_garden" #"aaa_demo1_putin_fail_light"
scene_file_name = "scene_file_0"
folder_path  = os.path.join(DIR.parent, "tasks")
json_path = os.path.join(DIR.parent, "tasks", task_name, scene_file_name)

simulator = Simulator(headless=True)


#################################### 采样一 ###########################################
# output_json_path = simulator.sample_custom_task(task_name="task6", scene_name="Rs_int")

# # 将生成的文件拷贝到 DIR/../btgym/assets/my_tasks/{task_name}/problem0.bddl 下作为 scene_file_0.json
# # 复制文件并重命名为scene_file_0.json
# target_path = os.path.join(DIR.parent, f"tasks/{task_name}", f'{scene_file_name}.json')
# shutil.copy2(json_path, target_path)
# print('已复制场景文件到:', target_path)

# 采样需要确定房间


#################################### 采样二 ###########################################
#Rs_int
simulator.load_custom_task(task_name=task_name, scene_name="Rs_garden",scene_file_name=scene_file_name, 
                                                folder_path=os.path.join(DIR.parent, f"tasks"),is_sample=True)
simulator.navigate_to_object("electric_refrigerator.n.01_1")
simulator.idle()
