from exps_bt_learning.tasks.task3.exec_lib._base.OGAction import OGAction

class PickUpPen(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["pen"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"IsNear({arg[0]})", "IsHandEmpty()"}
        info["add"] = {f"IsHolding({arg[0]})"}
        info["del_set"] = {"IsHandEmpty()"}
        info["cost"] = 1
        return info

    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]