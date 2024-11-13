from btgym.utils.path import ROOT_PATH

def get_activity_list():
    with open(f'{ROOT_PATH}/assets/task_names.txt', 'r') as f:
        task_names = f.read().splitlines()
    return task_names

