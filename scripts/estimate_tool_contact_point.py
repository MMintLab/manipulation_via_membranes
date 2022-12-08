#!/usr/bin/env python3
import rospy
import numpy as np
import sys
import argparse
from arc_utilities.tf2wrapper import TF2Wrapper
from bubble_control.bubble_contact_point_estimation.tool_contact_point_estimator import ToolContactPointEstimator
import tf.transformations as tr
from bubble_control.aux.load_confs import load_plane_params
import moveit_commander
from moveit_commander.conversions import pose_to_list

from geometry_msgs.msg import TransformStamped, Pose
from moveit_msgs.msg import AttachedCollisionObject, PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive

def estimate_tool_contact_point(plane_pose=None, current_scene=None, plane_size=(1.0, 1.0, 0.02, 0)):
    # plane_size: (dim_x, dim_y, dim_z_minus, dim_z_plus)
    if plane_pose is not None:
        ref_frame = 'med_base'
        plane_frame_name = 'plane_frame'
        try:
            rospy.init_node('contact_point_estimator')
        except (rospy.exceptions.ROSInitException, rospy.exceptions.ROSException):
            pass
        tf2_listener = TF2Wrapper()
        # create the transform
        plane_pos, plane_quat = np.split(plane_pose, [3])
        tf2_listener.send_transform(plane_pos, plane_quat, ref_frame, plane_frame_name, is_static=True)
        rospy.sleep(2.0)
        plane_object = _get_collision_object(dim_x=plane_size[0], dim_y=plane_size[1], thickness=plane_size[2], thick_plus=plane_size[3])
        current_scene.add_object(plane_object)
        rospy.sleep(2.0)
        tcpe = ToolContactPointEstimator(plane_frame=plane_frame_name)
    else:
        tcpe = ToolContactPointEstimator()

def _get_collision_object(dim_x=1.0, dim_y=1.0, thickness=0.02, thick_plus=0.0):
    co = CollisionObject()
    co.header.frame_id = '/plane_frame'
    co.id = 'plane_object'
    co.operation = co.ADD
    # co.operation = co.REMOVE
    # Define a the plane as a box:
    plane_sp = SolidPrimitive()
    plane_sp.type = plane_sp.BOX
    plane_sp.dimensions = [dim_x, dim_y, thickness+thick_plus]  # y is extended to prevent undesired motions
    plane_pose = Pose()
    plane_pose.position.x = 0
    plane_pose.position.y = 0
    plane_pose.position.z = -thickness * 0.5
    plane_pose.orientation.x = 0.0
    plane_pose.orientation.y = 0.0
    plane_pose.orientation.z = 0.0
    plane_pose.orientation.w = 1.0
    co.primitives.append(plane_sp)
    co.primitive_poses.append(plane_pose)
    return co

if __name__ == '__main__':
    default_pos = np.zeros(3)
    default_ori = np.array([0, 0, 0, 1])
    plane_params = load_plane_params()

    parser = argparse.ArgumentParser('Tool Contact Point Estimation')
    parser.add_argument('--pos', nargs=3, type=float, default=default_pos)
    parser.add_argument('--ori', nargs='+', type=float, default=default_ori)
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--width', type=float, default=1.0)
    parser.add_argument('--height', type=float, default=1.0)
    parser.add_argument('--thick', type=float, default=0.02)
    parser.add_argument('--thick_plus', type=float, default=0)

    args = parser.parse_args()
    pos = np.asarray(args.pos)
    ori = np.asarray(args.ori)
    plane_size = [args.width, args.height, args.thick, args.thick_plus]

    config_name = args.config
    if config_name != '':
        print('-- Loading plane configuration {} --'.format(config_name))
        plane_pose_raw = plane_params[config_name]['pose']
        plane_size = plane_params[config_name]['size']
        plane_pose = np.asarray(plane_pose_raw)
    else:
        # Use the ori, pos provided
        if len(ori) == 3:
            # euler angles (otherwise quaternion)
            ori = tr.quaternion_from_euler(ori[0], ori[1], ori[2])
        plane_pose = np.concatenate([pos, ori])
        print('Plane Pose:', plane_pose)

    rospy.init_node('column_collision_node', anonymous=True, disable_signals=True)
    moveit_commander.roscpp_initialize(sys.argv)
    current_scene = moveit_commander.PlanningSceneInterface(ns='med', synchronous=True)

    try:
        estimate_tool_contact_point(plane_pose=plane_pose, current_scene=current_scene, plane_size=plane_size)

    finally:
        current_scene.remove_world_object('plane_object')