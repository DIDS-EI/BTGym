
import json
import os

post_fix = '_0_0_template.json'
path = '/home/cxl/code_external/OmniGibson/omnigibson/data/og_dataset/scenes/'

def get_task_scene_map():
    task_scene_map = {}

    for scene_name in os.listdir(path):
        json_path = f'{path}/{scene_name}/json'
        for file_name in os.listdir(json_path):

            if file_name.endswith(post_fix):
                task_name = file_name[len(scene_name)+6:-len(post_fix)]
                if task_name not in task_scene_map:
                    task_scene_map[task_name] = []
                task_scene_map[task_name].append(scene_name)

    json.dump(task_scene_map, open('/home/cxl/code/BTGym/btgym/assets/task_to_scenes.json', 'w'), indent=4)




if __name__ == '__main__':
    for scene_name in os.listdir(path):
        json_path = f'{path}/{scene_name}/json'
        for file_name in os.listdir(json_path):
            task = json.load(open(f'{json_path}/{file_name}', 'r'))
            robot = task['objects_info']['init_info'].get('robot0', None)
            print(robot)