from exps_bt_learning.tasks.task5.exec_lib._base.OGAction import OGAction

class ToggleOn(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["oven"]

    def __init__(self, *args):
        super().__init__(*args)
        self.target_obj = self.args[0]

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]={"IsClosed({arg[0]})"}
        info["add"]={f"IsToggledOn({arg[0]})"}
        info["del_set"] = set()
        info["cost"] = 1
        return info

    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])