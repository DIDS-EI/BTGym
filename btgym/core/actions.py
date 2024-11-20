from btgym.core.simulator import Simulator
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitiveSet, StarterSemanticActionPrimitives
from btgym.utils.logger import log
import math
import omnigibson.utils.transform_utils as T
import torch as th
import random
from omnigibson.action_primitives.starter_semantic_action_primitives import indented_print\
    ,ActionPrimitiveError,m,detect_robot_collision_in_sim,multi_dim_linspace
from omnigibson.action_primitives.starter_semantic_action_primitives import PlanningContext

m.GRASP_APPROACH_DISTANCE = 0.3
m.MAX_STEPS_FOR_GRASP_OR_RELEASE = 50
m.MAX_STEPS_FOR_HAND_MOVE_JOINT = 100

def get_grasp_poses_for_object_sticky(target_obj):
    """
    Obtain a grasp pose for an object from top down, to be used with sticky grasping.

    Args:
        target_object (StatefulObject): Object to get a grasp pose for

    Returns:
        List of grasp candidates, where each grasp candidate is a tuple containing the grasp pose and the approach direction.
    """
    bbox_center_in_world, bbox_quat_in_world, bbox_extent_in_base_frame, _ = target_obj.get_base_aligned_bbox(
        visual=False
    )

    grasp_center_pos = bbox_center_in_world + th.tensor([0, 0, th.max(bbox_extent_in_base_frame) + 0.05])
    towards_object_in_world_frame = bbox_center_in_world - grasp_center_pos
    towards_object_in_world_frame /= th.norm(towards_object_in_world_frame)

    grasp_quat = T.euler2quat(th.tensor([0, math.pi / 2, 0], dtype=th.float32))

    grasp_pose = (grasp_center_pos, grasp_quat)
    grasp_candidate = [(grasp_pose, towards_object_in_world_frame)]

    return grasp_candidate



class ActionPrimitives(StarterSemanticActionPrimitives):
    def __init__(self, simulator:Simulator):
        self.simulator = simulator
        self.og_sim = simulator.og_sim
        super().__init__(self.og_sim, enable_head_tracking=False)

    def _navigate_to_obj(self, obj, pose_on_obj=None, **kwargs):
        """
        Yields action to navigate the robot to be in range of the pose

        Args:
            obj (StatefulObject): object to be in range of
            pose_on_obj (Iterable): (pos, quat) pose

        Returns:
            th.tensor or None: Action array for one step for the robot to navigate in range or None if it is done navigating
        """
        pose = self._sample_pose_near_object(obj, pose_on_obj=pose_on_obj, **kwargs)
        x,y,yaw = pose
        pose = (th.tensor([x, y, 0.0], dtype=th.float32), T.euler2quat(th.tensor([0, 0, yaw], dtype=th.float32)))

        self.robot.set_position_orientation(pose[0],pose[1])
        # 计算摄像机位置 - 在pose[0]前方2米
        forward_dir = T.quat2mat(pose[1])[:3,:3] @ th.tensor([1,0,0],dtype=th.float32) 
        camera_pos = pose[0] + 1.0 * forward_dir
        camera_pos[2] = 1.5 # 设置摄像机高度
        # 计算摄像机朝向 - 看向机器人
        # camera_dir = pose[0] - camera_pos
        # 反向四元数
        camera_quat = T.quat_inverse(pose[1])

        self.simulator.set_viewer_camera_pose(camera_pos,camera_quat)
        yield self._empty_action()



    def _grasp(self, obj):
        """
        Yields action for the robot to navigate to object if needed, then to grasp it

        Args:
            StatefulObject: Object for robot to grasp

        Returns:
            th.tensor or None: Action array for one step for the robot to grasp or None if grasp completed
        """
        # Update the tracking to track the object.
        self._tracking_object = obj

        # Don't do anything if the object is already grasped.
        obj_in_hand = self._get_obj_in_hand()
        if obj_in_hand is not None:
            if obj_in_hand == obj:
                return
            else:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                    "Cannot grasp when your hand is already full",
                    {"target object": obj.name, "object currently in hand": obj_in_hand.name},
                )

        # Open the hand first
        indented_print("Opening hand before grasping")
        yield from self._execute_release()

        # Allow grasping from suboptimal extents if we've tried enough times.
        indented_print("Sampling grasp pose")
        grasp_poses = get_grasp_poses_for_object_sticky(obj)
        grasp_pose, object_direction = random.choice(grasp_poses)

        # Prepare data for the approach later.
        approach_pos = grasp_pose[0] + object_direction * m.GRASP_APPROACH_DISTANCE
        approach_pose = (approach_pos, grasp_pose[1])

        # If the grasp pose is too far, navigate.
        indented_print("Navigating to grasp pose if needed")
        # yield from self._navigate_if_needed(obj, pose_on_obj=grasp_pose)
        yield from self._navigate_to_obj(obj, pose_on_obj=grasp_pose)

        indented_print("Moving hand to grasp pose")
        yield from self._move_hand(grasp_pose)

        # We can pre-grasp in sticky grasping mode.
        indented_print("Pregrasp squeeze")
        yield from self._execute_grasp()

        # Since the grasp pose is slightly off the object, we want to move towards the object, around 5cm.
        # It's okay if we can't go all the way because we run into the object.
        indented_print("Performing grasp approach")
        yield from self._move_hand_linearly_cartesian(approach_pose, stop_on_contact=True)
        # yield from self._move_hand_linearly_cartesian(approach_pose, stop_on_contact=True)

        # Step once to update
        empty_action = self._empty_action()
        yield self._postprocess_action(empty_action)

        indented_print("Checking grasp")
        if self._get_obj_in_hand() is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "Grasp completed, but no object detected in hand after executing grasp",
                {"target object": obj.name},
            )

        indented_print("Moving hand back")
        yield from self._reset_hand()

        indented_print("Done with grasp")

        if self._get_obj_in_hand() != obj:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "An unexpected object was detected in hand after executing grasp. Consider releasing it",
                {"expected object": obj.name, "actual object": self._get_obj_in_hand().name},
            )


    def _move_hand_linearly_cartesian(
        self, target_pose, stop_on_contact=False, ignore_failure=False, stop_if_stuck=False
    ):
        """
        Yields action for the robot to move its arm to reach the specified target pose by moving the eef along a line in cartesian
        space from its current pose

        Args:
            target_pose (Iterable of array): Position and orientation arrays in an iterable for pose
            stop_on_contact (boolean): Determines whether to stop move once an object is hit
            ignore_failure (boolean): Determines whether to throw error for not reaching final joint positions

        Returns:
            th.tensor or None: Action array for one step for the robot to move arm or None if its at the target pose
        """
        # To make sure that this happens in a roughly linear fashion, we will divide the trajectory
        # into 1cm-long pieces
        start_pos, start_orn = self.robot.eef_links[self.arm].get_position_orientation()
        travel_distance = th.norm(target_pose[0] - start_pos)
        num_poses = int(
            th.max(th.tensor([2, int(travel_distance / m.MAX_CARTESIAN_HAND_STEP) + 1], dtype=th.float32)).item()
        )
        pos_waypoints = multi_dim_linspace(start_pos, target_pose[0], num_poses)

        # Also interpolate the rotations
        t_values = th.linspace(0, 1, num_poses)
        quat_waypoints = [T.quat_slerp(start_orn, target_pose[1], t) for t in t_values]

        controller_config = self.robot._controller_config["arm_" + self.arm]
        if controller_config["name"] == "InverseKinematicsController":
            waypoints = list(zip(pos_waypoints, quat_waypoints))

            for i, waypoint in enumerate(waypoints):
                if i < len(waypoints) - 1:
                    yield from self._move_hand_direct_ik(
                        waypoint,
                        stop_on_contact=stop_on_contact,
                        ignore_failure=ignore_failure,
                        stop_if_stuck=stop_if_stuck,
                    )
                else:
                    yield from self._move_hand_direct_ik(
                        waypoints[-1],
                        pos_thresh=0.01,
                        ori_thresh=0.1,
                        stop_on_contact=stop_on_contact,
                        ignore_failure=ignore_failure,
                        stop_if_stuck=stop_if_stuck,
                    )

                # Also decide if we can stop early.
                current_pos, current_orn = self.robot.eef_links[self.arm].get_position_orientation()
                pos_diff = th.norm(current_pos - target_pose[0])
                orn_diff = T.get_orientation_diff_in_radian(target_pose[1], current_orn).item()
                if pos_diff < m.HAND_DIST_THRESHOLD and orn_diff < th.deg2rad(th.tensor([0.1])).item():
                    return

                if stop_on_contact and detect_robot_collision_in_sim(self.robot, ignore_obj_in_hand=False):
                    return

            if not ignore_failure:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.EXECUTION_ERROR,
                    "Your hand was obstructed from moving to the desired world position",
                )
        else:
            collision_counts = 0
            while True:
                if detect_robot_collision_in_sim(self.robot, ignore_obj_in_hand=False):
                    collision_counts += 1
                    if collision_counts > 3:
                        break
                else:
                    collision_counts = 0

                current_pos, current_orn = self.robot.eef_links[self.arm].get_position_orientation()
                target_pos = current_pos - th.tensor([0,0,0.02])
                joint_pos = self._convert_cartesian_to_joint_space((target_pos, current_orn))
                yield from self._move_hand_direct_joint(
                    joint_pos, stop_on_contact=stop_on_contact, ignore_failure=ignore_failure
                )
            

    def _sample_pose_near_object(self, obj, pose_on_obj=None, **kwargs):
        """
        Returns a 2d pose for the robot within in the range of the object and where the robot is not in collision with anything

        Args:
            obj (StatefulObject): Object to sample a 2d pose near
            pose_on_obj (Iterable of arrays or None): The pose to sample near

        Returns:
            2-tuple:
                - 3-array: (x,y,z) Position in the world frame
                - 4-array: (x,y,z,w) Quaternion orientation in the world frame
        """
        with PlanningContext(self.env, self.robot, self.robot_copy, "simplified") as context:
            for _ in range(m.MAX_ATTEMPTS_FOR_SAMPLING_POSE_NEAR_OBJECT):
                if pose_on_obj is None:
                    pos_on_obj = self._sample_position_on_aabb_side(obj)
                    pose_on_obj = [pos_on_obj, th.tensor([0, 0, 0, 1])]

                distance_lo, distance_hi = 0.2, 2
                distance = (th.rand(1) * (distance_hi - distance_lo) + distance_lo).item()
                yaw_lo, yaw_hi = -math.pi, math.pi
                yaw = th.rand(1) * (yaw_hi - yaw_lo) + yaw_lo
                avg_arm_workspace_range = th.mean(self.robot.arm_workspace_range[self.arm])
                pose_2d = th.cat(
                    [
                        pose_on_obj[0][0] + distance * th.cos(yaw),
                        pose_on_obj[0][1] + distance * th.sin(yaw),
                        yaw + math.pi - avg_arm_workspace_range,
                    ]
                )
                # Check room
                obj_rooms = (
                    obj.in_rooms
                    if obj.in_rooms
                    else [self.env.scene._seg_map.get_room_instance_by_point(pose_on_obj[0][:2])]
                )
                if self.env.scene._seg_map.get_room_instance_by_point(pose_2d[:2]) not in obj_rooms:
                    indented_print("Candidate position is in the wrong room.")
                    continue

                if not self._test_pose(pose_2d, context, pose_on_obj=pose_on_obj, **kwargs):
                    continue

                indented_print("Found valid position near object.")
                return pose_2d

            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.SAMPLING_ERROR,
                "Could not find valid position near object.",
                {
                    "target object": obj.name,
                    "target pos": obj.get_position_orientation()[0],
                    "pose on target": pose_on_obj,
                },
            )

class Action:
    def __init__(self,object_name):
        self.object_name = object_name
        self.object = None
        self.is_stoped = False

    def pre_start(self,action_primitives:ActionPrimitives, simulator:Simulator):
        self.object = simulator.og_sim.task.object_scope[self.object_name]
        # self.object = simulator.scene.object_registry("name", self.object_name)
        self.action_primitives = action_primitives
        self.simulator = simulator
        log(f"Action Start: {self.__class__.__name__}({self.object_name})")
        self.check_stop()

    def start(self,action_primitives:ActionPrimitives, simulator:Simulator):
        self.pre_start(action_primitives, simulator)
        self._start()

    def check_stop(self):
        self.is_stoped = False

    def step(self):
        try:
            action = next(self.action_control_generator)
            if action is not None:
                self.simulator.add_control(action)
            else:
                log('Action is None!!')
        except Exception as e:
            log(f"Action Error: {self.__class__.__name__}({self.object_name}) ==> {e}")
            self.start(self.action_primitives, self.simulator)

            # self.check_stop()
            # if self.is_stoped:
            #     return
            # else:
            #     log(f"Action Error: {self.__class__.__name__}({self.object_name}) ==> {e}")
            #     self.start(self.action_primitives, self.simulator)

            #     if self.action_primitives._get_obj_in_hand() is None:
            #         log(f"Action Error: {self.__class__.__name__}({self.object_name}) ==> {e}")
            # else:
            #     log(f"Action Error: {self.__class__.__name__}({self.object_name}) ==> {e}")
            #     self.start(self.action_primitives, self.simulator)
            # self.simulator.reset()


class Grasp(Action):
    def __init__(self, object_name):
        super().__init__(object_name)

    def _start(self):
        self.action_control_generator = self.action_primitives.apply_ref(
            StarterSemanticActionPrimitiveSet.GRASP, self.object,attempts=1)

    def check_stop(self):
        obj_in_hand = self.action_primitives._get_obj_in_hand()
        if obj_in_hand is not None:
            log(f"Object in hand: {obj_in_hand.name}")
            log(f"Action Stoped: {self.__class__.__name__}({self.object_name})")
            self.is_stoped = True


class Release(Action):
    def __init__(self, object_name):
        super().__init__(object_name)

    def _start(self):
        self.action_control_generator = self.action_primitives.apply_ref(
            StarterSemanticActionPrimitiveSet.RELEASE, self.object,attempts=1)

    def check_stop(self):
        obj_in_hand = self.action_primitives._get_obj_in_hand()
        if obj_in_hand is None:
            log(f"No object in hand.")
            log(f"Action Stoped: {self.__class__.__name__}({self.object_name})")
            self.is_stoped = True


class PlaceOnTop(Action):
    def __init__(self, object_name):
        super().__init__(object_name)

    def _start(self):
        self.action_control_generator = self.action_primitives.apply_ref(
            StarterSemanticActionPrimitiveSet.PLACE_ON_TOP, self.object,attempts=1)

    def check_stop(self):
        obj_in_hand = self.action_primitives._get_obj_in_hand()
        if obj_in_hand is None:
            log(f"No object in hand.")
            log(f"Action Stoped: {self.__class__.__name__}({self.object_name})")
            self.is_stoped = True


class PlaceInside(Action):
    def __init__(self, object_name):
        super().__init__(object_name)

    def _start(self):
        self.action_control_generator = self.action_primitives.apply_ref(
            StarterSemanticActionPrimitiveSet.PLACE_INSIDE, self.object,attempts=1)

    def check_stop(self):
        obj_in_hand = self.action_primitives._get_obj_in_hand()
        if obj_in_hand is None:
            log(f"No object in hand.")
            log(f"Action Stoped: {self.__class__.__name__}({self.object_name})")
            self.is_stoped = True


class Open(Action):
    def __init__(self, object_name):
        super().__init__(object_name)

    def _start(self):
        self.action_control_generator = self.action_primitives.apply_ref(
            StarterSemanticActionPrimitiveSet.OPEN, self.object,attempts=1)

class Close(Action):
    def __init__(self, object_name):
        super().__init__(object_name)

    def _start(self):
        self.action_control_generator = self.action_primitives.apply_ref(
            StarterSemanticActionPrimitiveSet.CLOSE, self.object,attempts=1)

class NavigateTo(Action):
    def __init__(self, object_name):
        super().__init__(object_name)

    def _start(self):
        self.action_control_generator = self.action_primitives.apply_ref(
            StarterSemanticActionPrimitiveSet.NAVIGATE_TO, self.object,attempts=1)

class ToggleOn(Action):
    def __init__(self, object_name):
        super().__init__(object_name)

    def _start(self):
        self.action_control_generator = self.action_primitives.apply_ref(
            StarterSemanticActionPrimitiveSet.TOGGLE_ON, self.object,attempts=1)

class ToggleOff(Action):
    def __init__(self, object_name):
        super().__init__(object_name)
        
    def _start(self):
        self.action_control_generator = self.action_primitives.apply_ref(
            StarterSemanticActionPrimitiveSet.TOGGLE_OFF, self.object,attempts=1)


action_map = {
    "grasp": Grasp,
    "release": Release,
    "place_on_top": PlaceOnTop,
    "place_inside": PlaceInside,
    "open": Open,
    "close": Close, 
    "navigate_to": NavigateTo,
    "toggle_on": ToggleOn,
    "toggle_off": ToggleOff,
}
