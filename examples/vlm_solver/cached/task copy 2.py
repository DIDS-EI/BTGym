
def do_task(env):
    """reorient the white pen and drop it upright into the black pen holder"""
    pen_id = 5
    env.grasp_object(pen_id)
    pose_target = env.get_object_pose_target(
        subgoal_text="move the pen upright above the pencil holder",
        object_id=pen_id)
    env.reach_pose(pose_target)
    env.open_gripper()
