from btgym.bt.base_nodes import Condition
from btgym.bt import Status
from btgym.cores.gridenv.minigrid_computation_env.base.WareHouseCondition import WareHouseCondition



class VHCondition(WareHouseCondition):
    can_be_expanded = True
    num_args = 1

