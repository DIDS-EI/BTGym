from exps_bt_learning.tasks.task1.exec_lib._base.OGAction import OGAction

class PickUpObject(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["apple"]

    def __init__(self, *args):
        super().__init__(*args)
        self.obj = self.args[0]

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]={f"On({arg[0]},{place})" for place in cls.valid_args}
        info["add"]={f"Holding({arg[0]})"}
        info["del_set"] = {f"On({arg[0]},{place})" for place in cls.valid_args}
        info["cost"] = 1
        return info

    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]