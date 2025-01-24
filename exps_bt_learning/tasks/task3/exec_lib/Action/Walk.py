from exps_bt_learning.tasks.task3.exec_lib._base.OGAction import OGAction

class Walk(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["coffee_cup", "table"]

    def __init__(self, *args):
        super().__init__(*args)
        self.target_obj = self.args[0]

    @classmethod
    def get_info(cls, *args):
        info = {}
        info["pre"] = set()
        info["add"] = {f"IsNear({args[0]})"}
        info["del_set"] = {f'IsNear({place})' for place in cls.valid_args if place != args[0]}
        info["cost"] = 1
        return info

    def change_condition_set(self):
        self.agent.condition_set |= self.info["add"]
        self.agent.condition_set -= self.info["del_set"]