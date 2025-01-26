import os
DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from PIL import Image
from btgym.behavior_tree.behavior_libs import ExecBehaviorLibrary
from btgym.algos.bt_planning.main_interface import collect_action_nodes

# 导入当前状态图片
image_path = os.path.join(DIR,"a_exp3/camera_putin_cross_level.png")
# image = Image.open(image_path)


# 获取动作
task_id = 2
task_name = f"task{task_id}"
bddl_file = os.path.join(DIR,f"tasks/{task_name}/problem0.bddl")
behavior_lib_path = os.path.join(DIR,f"tasks/{task_name}/exec_lib")  # os.path.join(DIR,"../exec_lib")
output_dir = os.path.join(DIR,f"tasks/{task_name}/bt.btml")

behavior_lib = ExecBehaviorLibrary(behavior_lib_path)
actions = collect_action_nodes(behavior_lib)


# for action in actions:
#     print(action.name)
#     print(action.pre)
#     print(action.add)
#     print(action.del_set)
#     print("--------------------------------")
putin_actions = [action for action in actions if "PlaceIn" in action.name]
putin_actions = putin_actions[0]

putin_actions.name = "PlaceIn(apple,drawer)"
putin_actions.pre = {'IsHolding(apple)','In(apple,drawer)'}
putin_actions.add = {'In(apple,drawer)'}
putin_actions.del_set = {'IsHolding(apple)'}

# 写 prompt
prompt = f"The action model of {putin_actions.name}: Preconditions: {putin_actions.pre}, Add Effects: {putin_actions.add}, Delete Effects: {putin_actions.del_set}"  
molmo_prompt = f"Point out the spot inside the cabinet to place a pen."  

query_prompt = f"The action {putin_actions.name} keeps failing. Check the image and identify the cause and examine if the action model is incorrect."

prompt = f"{prompt}\n{molmo_prompt}\n{query_prompt}"


# 导入大模型
from btgym.llm.llm_gpt import LLM
llm = LLM()

# 获取大模型输出
response = llm.request_instruction(prompt, image_path=image_path)
print(response)


# The action model for PlaceIn(apple,drawer) indicates that the preconditions are 'IsHolding(apple)' and 'In(apple,drawer)'. However, this seems contradictory because if the apple is already in the drawer ('In(apple,drawer)'), you wouldn't need to place it there again.

# From the image, it appears that the drawer is closed. This could be why the action keeps failing, as the apple cannot be placed inside a closed drawer.

# To correct the action model, you should modify the preconditions to ensure the apple is not already in the drawer and that the drawer is open. A more appropriate set of preconditions might be:

# Preconditions: {'IsHolding(apple)', 'DrawerOpen(drawer)'}

# This would ensure that the apple is being held and that the drawer is open before attempting to place the apple inside. The add and delete effects can remain the same:

# Add Effects: {'In(apple,drawer)'}
# Delete Effects: {'IsHolding(apple)'}

# Additionally, you may need an action to open the drawer if it is not already open. For example:

# Action: OpenDrawer(drawer)
# Preconditions: {'DrawerClosed(drawer)'}
# Add Effects: {'DrawerOpen(drawer)'}
# Delete Effects: {'DrawerClosed(drawer)'}

# Ensure these conditions are met before attempting to place the apple in the drawer.