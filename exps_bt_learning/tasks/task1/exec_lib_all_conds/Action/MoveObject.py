from exps_bt_learning.tasks.task1.exec_lib._base.OGAction import OGAction

class MoveObject(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = ["apple", "coffee_table"]

    def __init__(self, *args):
        super().__init__(*args)
        self.obj = self.args[0]
        self.target_obj = self.args[1]

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]={f"On({arg[0]},floor)"}
        info["add"]={f"On({arg[0]},{arg[1]})"}
        info["del_set"] = {f"On({arg[0]},place)" for place in cls.valid_args if place != arg[1]}
        info["cost"] = 1
        return info

    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]