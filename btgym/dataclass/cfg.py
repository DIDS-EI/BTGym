import os
from dataclasses import dataclass

@dataclass
class cfg:
    # Constants
    ROOT_PATH = os.path.abspath(os.path.join(__file__, "../.."))
    ASSETS_PATH = os.path.join(ROOT_PATH, "assets")
    OUTPUTS_PATH = os.path.join(ROOT_PATH, "../outputs")
    TESTS_PATH = os.path.join(ROOT_PATH, "../tests")
    SUBGOAL_NET_PATH = os.path.join(ROOT_PATH, "../examples/training/subgoal_net.pth")

    ##############
    # Variables
    ##############

    llm_model = "gpt-4o"
    # llm_model = "claude"
    # llm_model = "claude-3-5-sonnet-20241022"
    llm_temperature = 1
    # llm_max_tokens = 2048

    llm_embedding_model = "text-embedding-3-small"


    task_folder = f'{ASSETS_PATH}/my_tasks'
    task_json_folder = f'{ASSETS_PATH}/my_tasks'
    task_name = 'test_task'
    scene_name = 'Rs_int'
    scene_file_name = 'scene_file_0'



os.makedirs(cfg.TESTS_PATH, exist_ok=True)
os.makedirs(cfg.OUTPUTS_PATH, exist_ok=True)
