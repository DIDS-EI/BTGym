from examples.BTLearning.exec_lib._base.OGAction import OGAction

class PutOn(OGAction):
    can_be_expanded = True
    num_args = 2
    # valid_args = OGAction.CAN_INTERACT

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *args):
        info = {}
        info["pre"] = {
            f"IsHolding(self, {args[0]})",  # 确保抓取对象
            f"IsClear({args[1]})",  # 确保要放置的位置是空闲的
            f"IsReachable(self, {args[1]})"  # 确保可以到达要放置的位置
        }
        info["add"] = {
            f"IsOn({args[0]}, {args[1]})"  # 更新对象位置信息
        }
        info["del_set"] = {
            f"IsHolding(self, {args[0]})"  # 不再抓取该对象
        }
        info["cost"] = 5  # 假设这个动作的费用是5
        return info

    def change_condition_set(self):
        self.agent.condition_set |= self.info["add"]
        self.agent.condition_set -= self.info["del_set"]