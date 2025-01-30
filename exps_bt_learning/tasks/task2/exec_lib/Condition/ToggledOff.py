from exps_bt_learning.tasks.task2.exec_lib._base.OGCondition import OGCondition

class ToggledOff(OGCondition):
    can_be_expanded = True
    num_args = 1
    valid_args = ["light1", "light2"]