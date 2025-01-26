import os
DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from exps_bt_learning.tools import parse_bddl
from exps_bt_learning.llm_generate_lib_func import llm_generate_behavior_lib
from exps_bt_learning.validate_bt_fun import validate_bt_fun





# 1. 设置任务
task_name = "task1"

bddl_file = os.path.join(DIR,f"tasks/{task_name}/problem0.bddl")
behavior_lib_path = os.path.join(DIR,f"tasks/{task_name}/exec_lib")  # os.path.join(DIR,"../exec_lib")


objects, start_state, goal = parse_bddl(bddl_file)
# 把 goal 转换为字符串
goal_str = ' '.join(goal)
print("objects:",objects)
print("start_state:",start_state)
print("goal_str:",goal_str)

start_state.update({'IsHandEmpty()'})
# goal_str = 'IsHolding(apple)'
goal_str = 'On(apple,coffee_table)'


# 编写 5 个任务，每个任务包含 objects, start_state, goal_str
#



# 2. 运行实验
total_try_times = 10
success_times = 0

# 新建 result 目录,已经结果csv
result_dir = os.path.join(DIR,"results")
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
dataframe_path = os.path.join(result_dir,f"{task_name}_success_rate.csv")
table_data = []

for i in range(total_try_times):
    print(f"try {i+1} times")
    # 1. 生成行为库
    llm_generate_behavior_lib(bddl_file=bddl_file,goal_str=goal_str,objects=objects,start_state=start_state,behavior_lib_path=behavior_lib_path)
    # 2. 验证行为库 
    error,bt = validate_bt_fun(behavior_lib_path=behavior_lib_path, goal_str=goal_str,cur_cond_set=start_state,output_dir=None)
    if error == 0:
        success_times += 1
        
    # 输出生成的动作库数量和条件库数量
    action_lib_num = len(os.listdir(os.path.join(behavior_lib_path,'Action')))
    condition_lib_num = len(os.listdir(os.path.join(behavior_lib_path,'Condition')))
    print(f"action lib num: {action_lib_num}")
    print(f"condition lib num: {condition_lib_num}")
    # 把每次的结果存入表格
    # 表格的列：任务名,尝试次数，成功次数，动作库数量，条件库数量，成功是否
    # 表格的行：每次尝试
    table_data.append([task_name,i+1,action_lib_num,condition_lib_num,not error])
  
# 输出成功率
# 用百分比表示,保留两位小数
# 再输出 成功率/总次数
print(f"success rate: {success_times/total_try_times*100:.2f}%") 
print(f"success rate/total try times: {success_times}/{total_try_times}") 

# 把表格数据写入csv文件,英文标题
df = pd.DataFrame(table_data, columns=['task_name', 'try_times', 'action_lib_num', 'condition_lib_num', 'success_or_not'])
df.to_csv(dataframe_path, index=False)


    
error,bt = validate_bt_fun(behavior_lib_path=behavior_lib_path, goal_str=goal_str,cur_cond_set=start_state,output_dir=None)
  
