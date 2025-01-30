from exps_bt_learning.tasks.task1.exec_lib._base.OGAction import OGAction

class EatObject(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["apple"]

    def __init__(self, *args):
        super().__init__(*args)
        self.obj = self.args[0]

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]={f"Is({arg[0]},fruit)"}
        info["add"]={f"Eaten({arg[0]})"}
        info["del_set"] = {f"Is({arg[0]},type)" for type in ["fruit", "vegetable"]}
        info["cost"] = 1
        return info

    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]