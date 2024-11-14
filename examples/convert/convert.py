from bddl.activity import Conditions 
from btgym import ROOT_PATH
import os

activity_definitions_path = f"{ROOT_PATH}/assets/activity_definitions"
nonstandard_tasks_path = f"{ROOT_PATH}/assets/nonstandard_tasks"

for task_name in os.listdir(activity_definitions_path):
    task_path = f"{activity_definitions_path}/{task_name}"
    with open(task_path+'/problem0.bddl', 'r') as f:
        content = f.read()
    if 'forn' in content or 'forpairs' in content:
        print(task_name)
        os.system(f"mv {task_path} {nonstandard_tasks_path}")
