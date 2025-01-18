# from btgym.behavior_tree.scene.scene import Scene
# from btgym.behavior_tree.behavior_tree.btml.btmlCompiler import load

import os
from btgym.behavior_tree.behavior_trees.BehaviorTree import BehaviorTree
from btgym.behavior_tree.utils.draw import render_dot_tree
from btgym.behavior_tree.behavior_libs import ExecBehaviorLibrary
import pathlib  
from btgym.llm.llm import LLM
import re

DIR = pathlib.Path(__file__).parent
btml_path = os.path.join(DIR, 'Default.bt')
behavior_lib_path = os.path.join(DIR, 'exec_lib')

behavior_lib = ExecBehaviorLibrary(behavior_lib_path)
bt = BehaviorTree(btml_path, behavior_lib)


bt.print()

llm = LLM()


def extract_code(answer,file_path):
    # 使用正则表达式提取代码
    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, answer, re.DOTALL)
    if match:
        answer = match.group(1).strip()
    else:
        raise ValueError("No code found in the answer")
    with open(file_path, "w") as f:
        f.write(answer)



answer = llm.request_instruction('''
生成一个行为树的节点，例如：
```python
from examples.BTLearning.exec_lib._base.RHSAction import RHSAction

class Close(RHSAction):
    can_be_expanded = True
    num_args = 1
    valid_args = RHSAction.CAN_OPEN

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]={f"IsOpen({arg[0]})",f"IsNear(self,{arg[0]})","IsLeftHandEmpty(self)"} 
        info["add"]={f"IsClose({arg[0]})"}
        info["del_set"] = {f"IsOpen({arg[0]})"}
        info["cost"] = 3
        return info


    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]
```

模仿上面的例子，生成一个抓取节点PutOn:
                             ''')

file_path = os.path.join(DIR, 'exec_lib', 'Action', 'PutOn.py')
extract_code(answer,file_path)

