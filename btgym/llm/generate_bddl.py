
from btgym.llm.llm import LLM
import json
import random
from btgym.dataclass.cfg import cfg
from omnigibson.macros import gm
import os

with open(f"{cfg.ASSETS_PATH}/scene_list.txt", "r") as f:
    scene_list = [line.strip() for line in f.readlines()]

task_list = os.listdir(f"{cfg.ASSETS_PATH}/activity_definitions")

def random_task_bddl():
    task_name = random.choice(task_list)
    with open(f"{cfg.ASSETS_PATH}/activity_definitions/{task_name}/problem0.bddl", "r") as f:
        task_bddl = f.read()
    return task_bddl

# scene_name = random.choice(scene_list)
scene_name = "Rs_int"
with open(f"{gm.DATASET_PATH}/scenes/{scene_name}/json/{scene_name}_best.json", "r") as f:
    scene_json = json.load(f)
object_info = scene_json['objects_info']['init_info']

room_dict = {}
object_set = set()
for objname, info_dict in object_info.items():
    room_list = info_dict['args']["in_rooms"]
    for room in room_list:
        if room not in room_dict:
            room_dict[room] = []
        room_dict[room].append(objname)

print(room_dict)


example_1 = random_task_bddl()
example_2 = random_task_bddl()

prompt = f"""
"scene infomation":{room_dict},
"example 1":
'''
{example_1}
'''
,
"example 2":
'''
{example_2}
'''
"your goal": "generate a bddl file, make sure 
the goal only contains 'ontop' predicate."
"""




exit()

# print(prompt)

message = [
    {
        "role": "system",
        "content": "you are a task generator for a robot. The outputed task should be in the format of BDDL."
    },
    {
        "role": "user",
        "content": prompt
    }
]

llm = LLM()
print(llm.request(message))