from btgym.planning.planning import run_fast_downward
from btgym.utils.path import ROOT_PATH
from bddl.activity import Conditions 
import tempfile
import shutil
import os


def plan_single_task(task_path, domain_path, task_name,sas_file=None, debug=True):
    if not os.path.exists(task_path):
        return 
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

    success_file = f"{ROOT_PATH}/../outputs/bddl_planning/success/{task_name}"
    failure_file = f"{ROOT_PATH}/../outputs/bddl_planning/failures/{task_name}"


    plan = run_fast_downward(temp_file_path, task_path, success_file, sas_file, debug)
    if not os.path.exists(success_file):
        shutil.copy(task_path, failure_file)
        
    os.remove(temp_file_path)

    # print(updated_content)
    # print(task_name)

    # with open(f"{ROOT_PATH}/../outputs/bddl_planning/{task_name}.txt", "w") as f:
    #     for step in plan:
    #         f.write(step + "\n")
    return plan


def plan_single_task_wrapper(args):
    return plan_single_task(*args)

def plan_multi_task(num_tasks, debug=True):
    # 创建成功和失败文件夹
    shutil.rmtree(f"{ROOT_PATH}/../outputs/bddl_planning/success", ignore_errors=True)
    shutil.rmtree(f"{ROOT_PATH}/../outputs/bddl_planning/failures", ignore_errors=True)
    os.makedirs(f"{ROOT_PATH}/../outputs/bddl_planning/success", exist_ok=True)
    os.makedirs(f"{ROOT_PATH}/../outputs/bddl_planning/failures", exist_ok=True)

    # 创建临时文件夹
    temp_dir = f"{ROOT_PATH}/../outputs/bddl_planning/temp"
    shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)


    domain_path = f"{ROOT_PATH}/planning/domain_omnigibson.bddl"


    # 从文件中读取任务列表
    with open(f"{ROOT_PATH}/assets/task_names.txt", "r") as file:
        task_list = file.readlines()
        task_list.sort()

    from tqdm import tqdm
    from multiprocessing import Pool, cpu_count

    # 创建进程池
    pool = Pool(processes=cpu_count())
    
    # 准备任务参数
    task_params = []
    for task_name_raw in task_list[:num_tasks]:
        task_name = task_name_raw.strip()
        task_path = f"{ROOT_PATH}/assets/activity_definitions/{task_name}/problem0.bddl"
        sas_file = f"{temp_dir}/{task_name}.sas"
        task_params.append((task_path, domain_path, task_name, sas_file, debug))

    # 使用imap显示实时进度
    from queue import Queue
    from threading import Thread
    
    # 创建结果队列
    result_queue = Queue()
    plan_results = []
    
    # 创建消费者线程来显示进度
    def consumer():
        with tqdm(total=num_tasks) as pbar:
            for _ in range(num_tasks):
                result = result_queue.get()
                plan_results.append(result)
                pbar.update(1)
                
    consumer_thread = Thread(target=consumer)
    consumer_thread.start()
    
    # 生产者 - 将结果放入队列
    for result in pool.imap(plan_single_task_wrapper, task_params):
        result_queue.put(result)
        
    consumer_thread.join()
        
    # 关闭进程池
    pool.close()
    pool.join()
    


if __name__ == "__main__":
    # task_name = "passing_out_drinks"
    # task_path = f"{ROOT_PATH}/assets/activity_definitions/{task_name}/problem0.bddl"
    # domain_path = f"{ROOT_PATH}/planning/domain_omnigibson.bddl"
    # plan_single_task(task_path, domain_path,task_name)
    # exit()


    plan_multi_task(1016, debug=False)
    
    success_count = len(os.listdir(f"{ROOT_PATH}/../outputs/bddl_planning/success"))
    failure_count = len(os.listdir(f"{ROOT_PATH}/../outputs/bddl_planning/failures"))
    print(f'''
Success count: {success_count},
Failure count: {failure_count},
Total: {success_count + failure_count},
Ratio: {success_count / (success_count + failure_count)}
          ''')


    # task_name = "assembling_furniture"
    # task_path = f"{ROOT_PATH}/assets/activity_definitions/{task_name}/problem0.bddl"
    # domain_path = f"{ROOT_PATH}/planning/domain_omnigibson.bddl"
    # plan_single_task(task_path, domain_path, task_name)