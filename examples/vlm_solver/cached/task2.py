def do_task(env):
    """grasp the pen"""
    env.grasp_pos(grasp_point)
    pen_release_pose = pencil_holder.get_pen_release_pose(pen)
    env.reach_pose(pen_release_pose)
    env.open_gripper()
