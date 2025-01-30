from exps_bt_learning.tasks.task5.exec_lib._base.OGAction import OGAction

class PutIn(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = ["chickenleg","oven"]

    def __init__(self, *args):
        super().__init__(*args)
        self.obj = self.args[0]
        self.target_obj = self.args[1]

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"] = {f"IsNear({arg[1]})"}
        info["add"] = {f"In({arg[0]},{arg[1]})"}
        info["del_set"] = set()
        info["cost"] = 1
        return info

    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])