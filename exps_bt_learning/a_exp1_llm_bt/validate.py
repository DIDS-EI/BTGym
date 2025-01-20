
import os
from btgym.behavior_tree.behavior_trees.BehaviorTree import BehaviorTree
from btgym.behavior_tree.utils.draw import render_dot_tree
from btgym.behavior_tree.behavior_libs import ExecBehaviorLibrary
from btgym.algos.bt_planning.main_interface import BTExpInterface
from btgym.algos.llm_client.tools import goal_transfer_str
import pathlib  
from btgym.llm.llm import LLM
import re
import time
DIR = pathlib.Path(__file__).parent


######################
# 1. 导入正确的 lib
######################
behavior_lib_path = os.path.join(DIR, '../exec_lib_demo/exec_lib')
behavior_lib = ExecBehaviorLibrary(behavior_lib_path)


######################
# 2. 运行规划算法,返回规划成功与否
######################
goal_str = "IsHolding(apple)"
goal_set = goal_transfer_str(goal_str)

cur_cond_set = {'IsHandEmpty()'}

algo = BTExpInterface(behavior_lib, cur_cond_set=cur_cond_set,
                      selected_algorithm="obtea")

start_time = time.time()
algo.process(goal_set)
end_time = time.time()
planning_time_total = end_time - start_time

time_limit_exceeded = algo.algo.time_limit_exceeded

ptml_string, cost, expanded_num = algo.post_process()
error, state, act_num, current_cost, record_act_ls,ticks = algo.execute_bt(goal_set[0], cur_cond_set, verbose=False)

print(f"\x1b[32m Goal:{goal_str} \n Executed {act_num} action steps\x1b[0m",
      "\x1b[31mERROR\x1b[0m" if error else "",
      "\x1b[31mTIMEOUT\x1b[0m" if time_limit_exceeded else "")
print("current_cost:", current_cost, "expanded_num:", expanded_num, "planning_time_total:", planning_time_total)


######################
# 3. 输出算法结果
######################
file_name = "tree"
file_path = os.path.join(DIR,f'./{file_name}.btml')
with open(file_path, 'w') as file:
    file.write(ptml_string)
# read and execute
bt = BehaviorTree(file_path, behavior_lib)
bt.print()
bt.draw(target_directory=os.path.join(DIR,""))



# 测试 BT 的正确性