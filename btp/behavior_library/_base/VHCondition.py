from obtp.bt.base_nodes import Condition
from obtp.bt import Status
from obtp.envs.gridenv.minigrid_computation_env.base.WareHouseCondition import WareHouseCondition



class VHCondition(WareHouseCondition):
    can_be_expanded = True
    num_args = 1

