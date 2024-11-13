from btgym.planning.planning import run_fast_downward
from btgym.utils.path import ROOT_PATH
from bddl.activity import Conditions 
import tempfile
import shutil
import os


def plan_single_task(task_path, domain_path, task_name):
    activity_definition = 0                         # the specific definition you want to use. As of BEHAVIOR100 2021, this should always be 0.
    simulator = "omnigibson"                        # this does not require an actual simulator, just a domain file (e.g. activity_definitions/domain_omnigibson.bddl). You can make your own if desired.

    conds = Conditions(task_name, activity_definition, simulator)

    type_set = set(conds.parsed_objects.keys())

    with open(domain_path, 'r') as file:
        content = file.read()

    updated_content = content.replace('$type_list', ' '.join(type_set))


    # 使用临时文件存储更新后的内容
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
        temp_file.write(updated_content)
        temp_file_path = temp_file.name


    plan = run_fast_downward(temp_file_path, task_path, f"{ROOT_PATH}/../outputs/bddl_planning/{task_name}")

    os.remove(temp_file_path)

    print(updated_content)
    print(task_name)
    print(plan)
    if plan!=None:
        print("Plan found:")
        for step in plan:
            print(step)
    else:
        print("No plan found.")
    return plan



def plan_multi_task(num_tasks):
    os.makedirs(f"{ROOT_PATH}/../outputs/bddl_planning", exist_ok=True)
    domain_path = f"{ROOT_PATH}/planning/domain_omnigibson.bddl"


    # 从文件中读取任务列表
    with open(f"{ROOT_PATH}/assets/task_names.txt", "r") as file:
        task_list = file.readlines()
        task_list.sort()

    plan_success_count = 1
    for task_name_raw in task_list[:num_tasks]:
        task_name = task_name_raw.strip()
        task_path = f"{ROOT_PATH}/assets/activity_definitions/{task_name.strip()}/problem0.bddl"
        plan_result = plan_single_task(task_path, domain_path, task_name)
        
        if plan_result != None:
            plan_success_count += 1

    print(f"Plan success count: {plan_success_count}")
    print(f"Plan success rate: {plan_success_count/num_tasks}")

if __name__ == "__main__":
    plan_multi_task(1016)