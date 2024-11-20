import os
import queue
import multiprocessing

from btgym.utils.logger import set_logger_entry
import numpy as np
import yaml

import omnigibson as og
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
from omnigibson.macros import gm
from btgym.core.controller import Controller
from btgym.core.communicator import generate_two_way_communicator_pair

from btgym.utils.logger import log

# Don't use GPU dynamics and use flatcache for performance boost
# gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = True


def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)


def create_controller(communicator):
    Controller(communicator)




class Env:
    """
    Demonstrates how to use the action primitives to pick and place an object in an empty scene.

    It loads Rs_int with a robot, and the robot picks and places a bottle of cologne.
    """

    def __init__(self):
        # Load the config
        config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
        config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

        # Update it to create a custom environment and run some actions
        config["scene"]["scene_model"] = "Rs_int"
        config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]
        config["objects"] = [
            {
                "type": "DatasetObject",
                "name": "cologne",
                "category": "bottle_of_cologne",
                "model": "lyipur",
                "position": [-0.3, -0.8, 0.5],
                "orientation": [0, 0, 0, 1],
            },
            {
                "type": "DatasetObject",
                "name": "table",
                "category": "breakfast_table",
                "model": "rjgmmy",
                "scale": [0.3, 0.3, 0.3],
                "position": [-0.7, 0.5, 0.2],
                "orientation": [0, 0, 0, 1],
            },
        ]

        # Load the environment
        self.sim = og.Environment(configs=config)
        self.scene = self.sim.scene
        self.robot = self.sim.robots[0]
        nope_action = np.zeros(self.robot.action_space.shape)
        # # Allow user to move camera more easily
        og.sim.enable_viewer_camera_teleoperation()


        controller_env_comm, ctrl_comm = generate_two_way_communicator_pair()
        ctrl_comm.set_owner(self)
        process = multiprocessing.Process(target=create_controller, args=(controller_env_comm,))
        process.start() 

        # self.do_task()
        
        while True:
            # 从队列中获取消息
            ctrl_comm.deal_function()

            self.sim.step(nope_action)

    def get_scene_name(self):
        return 'Merom_0_int'
        # return self.scene.scene_model
    
    def get_robot_pos(self):
        robot_pos = self.robot.get_position()
        return robot_pos

    def do_task(self):
        log("start do_task")
        controller = StarterSemanticActionPrimitives(self.sim, enable_head_tracking=False)

        # Grasp of cologne
        grasp_obj = self.scene.object_registry("name", "cologne")
        print("Executing controller")

        primitive_action = controller.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, grasp_obj)

        execute_controller(primitive_action, self.sim)
        print("Finished executing grasp")

        # Place cologne on another table
        print("Executing controller")
        table = self.scene.object_registry("name", "table")
        execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.PLACE_ON_TOP, table), self.sim)
        print("Finished executing place")




if __name__ == "__main__":
    set_logger_entry(__file__)
    env = Env()
    env.run()

