from exps_bt_learning.tasks.task4.exec_lib._base.OGCondition import OGCondition

class Closed(OGCondition):
    can_be_expanded = True
    num_args = 1
    valid_args = ["cabinet"]