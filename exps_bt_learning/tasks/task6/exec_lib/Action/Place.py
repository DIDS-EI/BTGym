from exps_bt_learning.tasks.task6.exec_lib._base.OGAction import OGAction

class Place(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = ["agent", "floor", "cake", "oven", "microwave", "yard_table"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *args):
        info = {}
        info["pre"] = {f"IsHolding({args[0]})", f"IsNear({args[1]})"}
        info["add"] = {f"On({args[0]}, {args[1]})", "IsHandEmpty()"}
        info["del_set"] = {f"IsHolding({args[0]})"}
        info["cost"] = 1
        return info

    def change_condition_set(self):
        self.agent.condition_set |= self.info["add"]
        self.agent.condition_set -= self.info["del_set"]