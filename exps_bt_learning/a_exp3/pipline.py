import os
DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from exps_bt_learning.tools import parse_bddl
from exps_bt_learning.llm_generate_lib_func import llm_generate_behavior_lib
from exps_bt_learning.validate_bt_fun import validate_bt_fun
from btgym.behavior_tree.behavior_libs import ExecBehaviorLibrary
from btgym.algos.bt_planning.main_interface import collect_action_nodes

task2name = {
    "task1":"PlaceApple",
    "task2":"PutInDrawer",
    "task3":"ActivateLights",
    "task4":"HomeRearrangement",
    "task5":"MealPreparation",
    "task6":"MealPreparation2"
}
task2objects = {
    "task1":['apple','coffee_table'],
    "task2":['pen','cabinet'],
    "task3":['light1','light2'],
    "task4":['apple','coffee_table','pen','cabinet'],
    "task5":['oven','chicken_leg','apple','coffee_table'],
    "task6":['apple','oven','drawer','coffee_table']
}
task2start_state = {
    "task1":{'IsHandEmpty()'},
    "task2":{'IsHandEmpty()','Closed(cabinet)'},
    "task3":{'IsHandEmpty()'},
    "task4":{'IsHandEmpty()','Closed(cabinet)','In(pen,cabinet)'},
    "task5":{'IsHandEmpty()','IsOpen(oven)'},
    "task6":{'IsHandEmpty()','Closed(drawer)','UnToggled(oven)'}
}
task2goal_str = {
    "task1":'On(apple,coffee_table)',
    "task2":'In(pen,cabinet)',
    "task3":'Activated(light1) & Activated(light2)',
    "task4":'On(pen,coffee_table) & Closed(cabinet) & In(apple,cabinet)', #In(apple,cabinet) & Closed(cabinet) & 
    "task5":'Closed(oven) & Activated(oven) & On(apple,coffee_table) & On(chicken_leg,coffee_table)', #& On(apple,coffee_table) & On(chicken_leg,coffee_table)
    "task6":'IsOpen(drawer) & On(apple,coffee_table) & Toggled(oven)'
}



task_id = 6
task_name = f"task{task_id}"

bddl_file = os.path.join(DIR,f"tasks/{task_name}/problem0.bddl")
behavior_lib_path = os.path.join(DIR,f"tasks/{task_name}/exec_lib")  # os.path.join(DIR,"../exec_lib")
output_dir = os.path.join(DIR,f"tasks/{task_name}/bt.btml")


objects, start_state, goal = parse_bddl(bddl_file)
objects = set(task2objects[task_name])
# 把 goal 转换为字符串
goal_str = ' '.join(goal)
print("objects:",objects)
print("start_state:",start_state)
print("goal_str:",goal_str)
start_state.update(task2start_state[task_name])
goal_str = task2goal_str[task_name]


# 1. 生成行为库
# llm_generate_behavior_lib(bddl_file=bddl_file,goal_str=goal_str,objects=objects,start_state=start_state,behavior_lib_path=behavior_lib_path)
# 2. 验证行为库 
print("Validate behavior lib...")
try:
    error,bt,expanded_num,act_num = validate_bt_fun(behavior_lib_path=behavior_lib_path, goal_str=goal_str,cur_cond_set=start_state,output_dir=output_dir)
except Exception as e:
    error=True
    act_num=-1
    expanded_num=-1
    print(f"error: {e}")
    
# # 输出生成的动作库数量和条件库数量
action_lib_num = len(os.listdir(os.path.join(behavior_lib_path,'Action')))
condition_lib_num = len(os.listdir(os.path.join(behavior_lib_path,'Condition')))
print(f"action lib num: {action_lib_num}")
print(f"condition lib num: {condition_lib_num}")



# 单元测试
# behavior_lib = ExecBehaviorLibrary(behavior_lib_path)
# actions = collect_action_nodes(behavior_lib)

# for action in actions:
#     print(action.name)
#     print(action.pre)
#     print(action.add)
#     print(action.del_set)
#     print("--------------------------------")
    
# action = actions[0]

# 为它生成采样场景
# PlaceIn 
# PlaceIn(pen,cabinet)
# {'IsHolding(pen)', 'IsNear(cabinet)'}
# {'IsHandEmpty()', 'In(pen,cabinet)'}
# {'IsHolding(pen)'}


# 2 才自洽

# 4个动作
# 1. 生成采样场景  5 
# 2. 生成采样场景   8
# 3. 生成采样场景   10 —————— 反馈 VLM ———— 修改了 putin 增加了 open
# 4. 生成采样场景   5


# 继续测试 putin 5
# 测试 open  6



# 上层： 2
# molomo： 39
# 跨层：3


