# from btgym.llm.llm import LLM
from btgym.llm.llm_gpt import LLM
import re
import pathlib
DIR = pathlib.Path(__file__).parent
import os
from exps_bt_learning.tools import parse_bddl,build_prompt,extract_code
from exps_bt_learning.validate_bt_fun import validate_bt_fun
from btgym.behavior_tree.behavior_libs import ExecBehaviorLibrary

########################
# 1. set input BDDL file 设置输入 BDDL 文件
########################
bddl_file = os.path.join(DIR,"problem0.bddl")
objects, start_state, goal = parse_bddl(bddl_file)
# 把 goal 转换为字符串
goal_str = ' '.join(goal)
print("objects:",objects)
print("start_state:",start_state)
print("goal_str:",goal_str)

start_state.update({'IsHandEmpty()'})
# goal_str = 'IsHolding(apple)'
goal_str = 'On(apple,coffee_table)'


behavior_lib_path = os.path.join(DIR,"../exec_lib")
# 先收集所有 lib
# behavior_lib = ExecBehaviorLibrary(behavior_lib_path)

########################
# 2. call llm to generate behavior lib 调用大模型生成行为库
########################
llm = LLM()
prompt = build_prompt(goal=goal_str,objects=objects)
# 蓝色打印
print("\033[94m",f"Requesting llm...","\033[0m")
answer = llm.request_instruction(prompt)
# 绿色打印
print("\033[92m",answer,"\033[0m")

# 提取所有 Python 代码块
# 把对应的 python 代码都提取到一个列表中,然后分别在对应的目录下创建动作类py文件
code_blocks_pattern = r"```python\n(.*?)```"
code_blocks = re.findall(code_blocks_pattern, answer, re.DOTALL)

# 解析每个代码块
class_pattern = r'class\s+(\w+)\s*\((\w+)\):'
for code in code_blocks:
    # 提取类定义
    matches = re.finditer(class_pattern, code) # 迭代器是一次性的
    for match in matches:
        class_name = match.group(1)
        base_class = match.group(2)
        
        # 确定类的类型
        if base_class == 'OGAction':
            class_type = "Action"
            import_statement = "from exps_bt_learning.exec_lib._base.OGAction import OGAction\n\n"
        elif base_class == 'OGCondition':
            class_type = "Condition"
            import_statement = "from exps_bt_learning.exec_lib._base.OGCondition import OGCondition\n\n"
        else:
            continue  # 如果基类不匹配，跳过这个类
        
        # 找到类定义的起始位置
        start_index = code.find(f"class {class_name}")
        
        # 找到下一个类定义的起始位置，或者代码块的结束位置
        end_index = code.find("class ", start_index + 1)
        if end_index == -1:
            end_index = len(code)
        
        # 提取单个类的代码
        single_class_code = code[start_index:end_index].strip()
        
        # 创建目录（如果不存在）
        class_dir = os.path.join(behavior_lib_path, class_type)
        os.makedirs(class_dir, exist_ok=True)
        
        # 写入 .py 文件，确保导入语句在文件开头
        file_path = os.path.join(class_dir, f"{class_name}.py")
        with open(file_path, "w") as f:
            f.write(import_statement + single_class_code)

        # 蓝色打印
        print("\033[94m",f"Written {class_name} to {file_path}","\033[0m")
    
########################
# 3. run planning algorithm and validate behavior lib 运行规划算法并验证行为库
########################
output_dir = os.path.join(DIR,"./tree.btml")
error,bt = validate_bt_fun(behavior_lib_path=behavior_lib_path, goal_str=goal_str,cur_cond_set=start_state,output_dir=output_dir)


