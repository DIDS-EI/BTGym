from exps_bt_learning.tasks.task5.exec_lib._base.OGCondition import OGCondition

class In(OGCondition):
    can_be_expanded = True
    num_args = 2


# Example usage
goal = "Closed(oven) & ToggledOn(oven) & On(apple,coffee_table) & In(chicken_leg,oven)"
objects = ["chicken_leg", "coffee_table", "agent", "apple", "electric_refrigerator", "floor", "oven", "microwave"]