from btgym.utils.path import ROOT_PATH
from btgym.utils import cfg
from btgym.utils.patches import apply_all_patches

apply_all_patches()

def get_activity_list():
    with open(f'{ROOT_PATH}/assets/task_names.txt', 'r') as f:
        task_names = f.read().splitlines()
    return task_names

