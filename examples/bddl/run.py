from btgym.planning import run_fast_downward
from btgym.utils.path import ROOT_PATH
#run_fast_downward("/home/cxl/code/BTGym/examples/pddl/domain.pddl", "/home/cxl/code/BTGym/examples/pddl/problem.pddl")

BDDL_PATH = "/home/cxl/code_external/bddl"
bddl_folder_path = f'{BDDL_PATH}/bddl/activity_definitions'
domain_file = f'{ROOT_PATH}/../examples/bddl/domain_omnigibson.bddl'


with open(f'{ROOT_PATH}/assets/task_names.txt', 'r') as f:
    task_name_list = f.read().splitlines()

task_name_list = sorted(task_name_list)

# for task_name in task_name_list[:20]:
#     print(task_name)
#     problem_file =  f'{ROOT_PATH}/../examples/bddl/problem0.bddl'
#     run_fast_downward(domain_file, problem_file,f'outputs/{task_name}')    

task_name = 'problem0'
problem_file =  f'{ROOT_PATH}/../examples/bddl/problem0.bddl'
plan = run_fast_downward(domain_file, problem_file,f'outputs/{task_name}')    
print(plan)