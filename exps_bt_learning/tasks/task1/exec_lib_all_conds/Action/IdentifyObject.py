from exps_bt_learning.tasks.task1.exec_lib._base.OGAction import OGAction

class IdentifyObject(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["apple"]

    def __init__(self, *args):
        super().__init__(*args)
        self.obj = self.args[0]

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]=set()
        info["add"]={f"Is({arg[0]},fruit)"}
        info["del_set"] = {f"Is(obj,type)" for obj in cls.valid_args if obj != arg[0] for type in ["fruit", "vegetable"]}
        info["cost"] = 1
        return info

    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]