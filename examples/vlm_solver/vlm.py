
import os
from btgym.llm.llm import LLM
from btgym.dataclass.cfg import cfg
import re
import importlib
import sys
# cfgs.llm_model = "claude-3-5-sonnet-20240620"

class VLM:
    def __init__(self):
        self.exp_output_path = os.path.join(cfg.OUTPUTS_PATH, "vlm")
        # 创建__init__.py文件
        init_file_path = os.path.join(self.exp_output_path, "__init__.py")
        if not os.path.exists(init_file_path):
            with open(init_file_path, "w") as f: pass
        
        os.makedirs(self.exp_output_path, exist_ok=True)
        if self.exp_output_path not in sys.path:
            sys.path.append(self.exp_output_path)

        self.llm = LLM()
        answer = self.llm.request("generate a python code to model the object 'pen', named Pen. including get_grasp_pose()")
        self.convert_answer_to_file(answer)

    def convert_answer_to_file(self, answer):
        answer_file_path = os.path.join(self.exp_output_path, "Pen.py")
        # 使用正则表达式提取代码
        pattern = r"```python\n(.*?)```"
        match = re.search(pattern, answer, re.DOTALL)
        if match:
            answer = match.group(1).strip()
        else:
            raise ValueError("No code found in the answer")
        with open(answer_file_path, "w") as f:
            f.write(answer)

    def run(self):
        pen = importlib.import_module("Pen").Pen()
        print(pen.get_grasp_pose())

if __name__ == "__main__":
    vlm = VLM()
    vlm.run()

