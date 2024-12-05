def do_task(env):
    """grasp the pen"""
    pen_obj, pencil_holder_obj = env.get_involved_object_names()
    pen = env.get_obj("pen_1")
    pencil_holder = env.get_obj("pencil_holder_1")
    grasp_point = [-0.15, -0.15, 0.72]
    env.grasp_pos(grasp_point)
    pen_release_pose = pencil_holder.get_pen_release_pose(pen)
    env.reach_pose(pen_release_pose)
    env.open_gripper()
