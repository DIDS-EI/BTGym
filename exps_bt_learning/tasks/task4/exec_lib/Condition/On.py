from exps_bt_learning.tasks.task4.exec_lib._base.OGCondition import OGCondition

class On(OGCondition):
    can_be_expanded = True
    num_args = 2
    valid_args = ["pen", "apple", "coffee_table"]