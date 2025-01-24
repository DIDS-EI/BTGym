import os
DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from exps_bt_learning.tools import parse_bddl
from exps_bt_learning.llm_generate_lib_func import llm_generate_behavior_lib
from exps_bt_learning.validate_bt_fun import validate_bt_fun

task2name = {
    "task1":"PlaceApple",
    "task2":"PutInDrawer",
    "task3":"ActivateLights",
    "task4":"HomeRearrangement",
    "task5":"MealPreparation"
}
task2objects = {
    "task1":['apple','coffee_table'],
    "task2":['pen','cabinet'],
    "task3":['light1','light2'],
    "task4":['apple','coffee_table','pen','cabinet'],
    "task5":['oven','chicken_leg','apple','coffee_table']
}
task2start_state = {
    "task1":{'IsHandEmpty()'},
    "task2":{'IsHandEmpty()','Closed(cabinet)'},
    "task3":{'IsHandEmpty()','ToggledOff(light1)','ToggledOn(light2)'},
    "task4":{'IsHandEmpty()','Open(cabinet)','In(pen,cabinet)'},
    "task5":{'IsHandEmpty()','IsOpen(oven)','ToggledOff(oven)'}
}
task2goal_str = {
    "task1":'On(apple,coffee_table)',
    "task2":'In(pen,cabinet)',
    "task3":'ToggledOn(light1) & ToggledOff(light2)',
    "task4":'On(pen,coffee_table) & Closed(cabinet) & On(apple,coffee_table)', #In(apple,cabinet) & Closed(cabinet) & 
    "task5":'Closed(oven) & ToggledOn(oven) & On(apple,coffee_table) & In(chicken_leg,oven)' #& On(apple,coffee_table) & On(chicken_leg,coffee_table)
}



for task_id in range(5,6):

    # 1. 设置任务
    task_name = f"task{task_id}"

    bddl_file = os.path.join(DIR,f"tasks/{task_name}/problem0.bddl")
    behavior_lib_path = os.path.join(DIR,f"tasks/{task_name}/exec_lib")  # os.path.join(DIR,"../exec_lib")
    output_dir = os.path.join(DIR,f"tasks/{task_name}/bt.btml")


    # objects, start_state, goal = parse_bddl(bddl_file)
    # goal_str = ' '.join(goal) # 把 goal 转换为字符串
    objects, start_state, _ = parse_bddl(bddl_file)
    
    # start_state = task2start_state[task_name]
    # objects = set(task2objects[task_name])
    # goal_str = task2goal_str[task_name]
    
    objects.update(task2objects[task_name])
    start_state.update(task2start_state[task_name])
    goal_str = task2goal_str[task_name]
    print("objects:",objects)
    print("start_state:",start_state)
    print("goal_str:",goal_str)


    # 编写 5 个任务，每个任务包含 objects, start_state, goal_str
    #



    # 2. 运行实验
    total_try_times = 10
    success_times = 0

    # 新建 result 目录,已经结果csv
    result_dir = os.path.join(DIR,"results")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    dataframe_path = os.path.join(result_dir,f"exp1_{task_name}_success_rate_{total_try_times}_NF_example=1.csv")
    table_data = []

    for i in range(total_try_times):
        print(f"try {i+1} times")
        # 1. 生成行为库
        llm_generate_behavior_lib(bddl_file=bddl_file,goal_str=goal_str,objects=objects,start_state=start_state,behavior_lib_path=behavior_lib_path)
        # 2. 验证行为库 
        print("Validate behavior lib...")
        try:
            error,bt,expanded_num,act_num = validate_bt_fun(behavior_lib_path=behavior_lib_path, goal_str=goal_str,cur_cond_set=start_state,output_dir=output_dir)
            if error == 0:
                success_times += 1
        except Exception as e:
            error=True
            act_num=-1
            expanded_num=-1
            print(f"error: {e}")
            
        # 输出生成的动作库数量和条件库数量
        action_lib_num = len(os.listdir(os.path.join(behavior_lib_path,'Action')))
        condition_lib_num = len(os.listdir(os.path.join(behavior_lib_path,'Condition')))
        print(f"action lib num: {action_lib_num}")
        print(f"condition lib num: {condition_lib_num}")
        # 把每次的结果存入表格
        # 表格的列：任务名,尝试次数，成功次数，动作库数量，条件库数量，成功是否
        # 表格的行：每次尝试
        table_data.append([task_name,i+1,action_lib_num,condition_lib_num,expanded_num,act_num,not error])
    
    # 输出成功率
    # 用百分比表示,保留两位小数
    # 再输出 成功率/总次数
    print(f"success rate: {success_times/total_try_times*100:.2f}%") 
    print(f"success rate/total try times: {success_times}/{total_try_times}")
    
    # 最后一列为平均值:action_lib_num的平均值,condition_lib_num的平均值,expanded_num的平均值,act_num的平均值
    # 平均值只需要计算 success 的次数
    action_lib_num_avg = sum([row[2] for row in table_data if row[6]]) / success_times
    condition_lib_num_avg = sum([row[3] for row in table_data if row[6]]) / success_times
    expanded_num_avg = sum([row[4] for row in table_data if row[6]]) / success_times
    act_num_avg = sum([row[5] for row in table_data if row[6]]) / success_times
    table_data.append([task_name,-1,action_lib_num_avg,condition_lib_num_avg,expanded_num_avg,act_num_avg,not error])

    # 把表格数据写入csv文件,英文标题
    df = pd.DataFrame(table_data, columns=['task_name', 'try_times', 'action_lib_num', 'condition_lib_num','expanded_num','act_num','success_or_not'])
    df.to_csv(dataframe_path, index=False)

print("done")
    
