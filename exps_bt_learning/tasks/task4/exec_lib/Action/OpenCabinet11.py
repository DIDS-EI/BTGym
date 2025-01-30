from exps_bt_learning.tasks.task4.exec_lib._base.OGAction import OGAction

class OpenCabinet11(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["cabinet"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"IsNear({arg[0]})"}
        info["add"] = {f"IsOpen({arg[0]})"}
        info["del_set"] = {f"IsClose({arg[0]})"}
        info["cost"] = 1
        return info

    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]