from exps_bt_learning.tasks.task5.exec_lib._base.OGAction import OGAction

class Close(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["oven"]

    def __init__(self, *args):
        super().__init__(*args)
        self.target_obj = self.args[0]

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"IsOpen({arg[0]})", f"IsNear({arg[0]})"}
        info["add"] = {f"Closed({arg[0]})"}
        info["del_set"] = {f"IsOpen({arg[0]})"}
        info["cost"] = 1
        return info

    def change_condition_set(self):
        self.agent.condition_set |= self.info["add"]
        self.agent.condition_set -= self.info["del_set"]