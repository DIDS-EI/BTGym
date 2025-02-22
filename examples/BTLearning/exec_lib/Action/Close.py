from examples.BTLearning.exec_lib._base.RHSAction import RHSAction

class Close(RHSAction):
    can_be_expanded = True
    num_args = 1
    valid_args = RHSAction.CAN_OPEN

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]={f"IsOpen({arg[0]})",f"IsNear(self,{arg[0]})","IsLeftHandEmpty(self)"} 
        info["add"]={f"IsClose({arg[0]})"}
        info["del_set"] = {f"IsOpen({arg[0]})"}
        info["cost"] = 3
        return info


    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]
