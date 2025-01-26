from exps_bt_learning.tasks.task2.exec_lib._base.OGCondition import OGCondition

class In(OGCondition):
    can_be_expanded = True
    num_args = 2


# Example usage
goal = "In(pen, cabinet)"
objects = ["pen", "cabinet"]