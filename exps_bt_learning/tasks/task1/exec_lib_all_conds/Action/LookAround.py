from exps_bt_learning.tasks.task1.exec_lib._base.OGAction import OGAction

class LookAround(OGAction):
    can_be_expanded = True
    num_args = 0
    valid_args = []

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]=set()
        info["add"]={"LookingAround()"}
        info["del_set"] = {"LookingAt(obj)" for obj in cls.valid_args}
        info["cost"] = 1
        return info

    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]