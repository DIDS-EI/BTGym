from exps_bt_learning.tasks.task3.exec_lib._base.OGAction import OGAction

class PlacePenInCabinet(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = ["pen", "cabinet"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"IsHolding({arg[0]})", f"IsOpen({arg[1]})", f"IsNear({arg[1]})"}
        info["add"] = {f"In({arg[0]},{arg[1]})", "IsHandEmpty()"}
        info["del_set"] = {f"IsHolding({arg[0]})"}
        info["cost"] = 1
        return info

    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]