from bddl.activity import Conditions 

import os

task_list = os.listdir("/home/cxl/code/BTP/activity_definitions")
with open('output.txt', 'w') as f:
    f.write('\n'.join(task_list))

# type_list = set()
# for task_name in task_list:
#     behavior_activity = task_name
#     activity_definition = 0                         # the specific definition you want to use. As of BEHAVIOR100 2021, this should always be 0.
#     simulator = "omnigibson"                        # this does not require an actual simulator, just a domain file (e.g. activity_definitions/domain_omnigibson.bddl). You can make your own if desired.

#     conds = Conditions(behavior_activity, activity_definition, simulator)

#     # You can now use the functions in bddl/activity.py to interact with the conds object. This generally requires a backend that's based on the simulator; in this case, you can use a stub backend. You can create something similar to, or directly use, the TrivialBackend, TrivialObject, TrivialSimulator, and various Trivial*Predicate classes found in bddl/bddl/trivial_backend.py.

#     type_list.update(set(conds.parsed_objects.keys()))

# # 将 type_list 的内容保存到文件中
# with open('output.txt', 'w') as f:
#     f.write(' '.join(type_list))

# # 如果仍然需要在控制台打印，可以保留这行

