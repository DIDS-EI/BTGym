import os
import re

def extract_types_from_file(file_path):
    types = set()
    with open(file_path, 'r') as file:
        content = file.read()
        # 使用正则表达式匹配类型定义
        matches = re.findall(r'(\w+\.\w+_\d+)\s*-\s*(\w+\.\w+)', content)
        for match in matches:
            types.add(match[1])
    return types

def collect_all_types(directory):
    all_types = set()
    for filename in os.listdir(directory):
        if filename.endswith('.bddl'):
            file_path = os.path.join(directory, filename)
            types = extract_types_from_file(file_path)
            all_types.update(types)
    return all_types

def generate_types_pddl(types):
    return f"(:types {' '.join(sorted(types))})"

# 使用示例
directory_path = 'activity_definitions'
all_types = collect_all_types(directory_path)
types_pddl = generate_types_pddl(all_types)
print(types_pddl)