from exps_bt_learning.tasks.task6.exec_lib._base.OGAction import OGAction

class Toggle(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["oven"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *args):
        info = {}
        info["pre"] = {f"IsNear({args[0]})"}
        info["add"] = {f"Toggled({args[0]})"}
        info["del_set"] = set()
        info["cost"] = 1
        return info

    def change_condition_set(self):
        self.agent.condition_set |= self.info["add"]
        self.agent.condition_set -= self.info["del_set"]