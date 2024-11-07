from pprint import pprint
from btgym.utils.path import ROOT_PATH
import subprocess

def run_fast_downward(domain_file, problem_file, output_file):
    # 定义 Fast Downward 的命令
    command = [f'{ROOT_PATH}/planning/downward/fast-downward.py',
                # '--plan-file', output_file,
                # '--overall-time-limit', '1',
                domain_file,
                problem_file,
                '--search', 'astar(lmcut())'
    ]

    # 使用 subprocess 运行命令
    result = subprocess.run(command, capture_output=True, text=True)

    pprint(result.stdout)
    pprint(result.stderr)

    # 检查是否成功运行
    if result.returncode != 0:
        print("Error running Fast Downward:")
        print(result.stderr)
        return None

    # 解析输出
    plan = []
    for line in result.stdout.splitlines():
        if line.startswith('step'):
            plan.append(line)

    return plan

if __name__ == "__main__":
    # 使用示例
    domain_file = "/home/cxl/code/BTGym/examples/domain.pddl"
    problem_file = "/home/cxl/code/BTGym/examples/problem.pddl"
    plan = run_fast_downward(domain_file, problem_file) 

    if plan:
        print("Plan found:")
        for step in plan:
            print(step)
    else:
        print("No plan found.")