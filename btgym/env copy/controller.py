import multiprocessing
import os
import time
import cv2
from omnigibson.utils.asset_utils import get_og_scene_path

from btgym.core.communicator import generate_one_way_communicator_pair, RequestCommunicator
from btgym.core.ui import UI
from btgym.utils.logger import log

def create_ui(*args):
    ui = UI(*args)
    return ui

class Controller:
    def __init__(self, communicator:RequestCommunicator):
        self.sim_comm = communicator
        self.sim_comm.set_owner(self)

        scene_name = self.sim_request("get_scene_name")
        log(f"Controller init scene_name: {scene_name}")
        self.info = {
            "scene_name": scene_name,
            "robot_pos": [0,0,0]
        }


        self.create_ui()   
        trav_map = self.show_scene_trav_map(scene_name)
        self.ui_func("set_image", trav_map)
        self.sim_func("do_task")
        self.run()


    def step(self):
        self.ui_receiver.deal_functions()
        
        self.info["robot_pos"] = self.sim_comm.call("get_robot_pos")
        self.ui_func("update_info", self.info)

    def create_ui(self):
        self.ui_sender, ctrl_ui_receiver = generate_one_way_communicator_pair()
        ui_ctrl_sender, self.ui_receiver = generate_one_way_communicator_pair()

        self.ui_receiver.set_owner(self)
        process = multiprocessing.Process(target=create_ui, args=(ui_ctrl_sender,ctrl_ui_receiver))
        process.start()


    def show_scene_trav_map(self,scene_name):
        trav_map = cv2.imread(os.path.join(get_og_scene_path(scene_name), "layout", "floor_trav_no_door_0.png"))
        return trav_map
    

    def sim_request(self,func_name,*args,**kwargs):
        self.sim_comm.request(func_name,*args,**kwargs)

    def sim_func(self,func_name,*args,**kwargs):
        log(f"sim_func: {func_name}")
        self.sim_comm.call(func_name,*args,**kwargs)

    def ui_func(self,func_name,*args,**kwargs):
        self.ui_sender.call(func_name,*args,**kwargs)

    def run(self):
        while True:
            self.step()
            time.sleep(0.01)


