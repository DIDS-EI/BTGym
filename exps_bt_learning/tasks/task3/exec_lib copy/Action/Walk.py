from exps_bt_learning.tasks.task3.exec_lib._base.OGAction import OGAction

class Walk(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["agent", "light1", "light2", "switch", "floor"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = set()
        info["add"] = {f"IsNear({arg[0]})"}
        info["del_set"] = {f"IsNear({place})" for place in cls.valid_args if place != arg[0]}
        info["cost"] = 1
        return info

    def change_condition_set(self):
        self.agent.condition_set |= self.info["add"]
        self.agent.condition_set -= self.info["del_set"]