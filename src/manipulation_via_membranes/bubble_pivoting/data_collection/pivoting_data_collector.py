#! /usr/bin/env python
import os
import pdb
import sys
import numpy as np
import threading
import copy
import rospy
import tf
import tf.transformations as tr
import argparse

from manipulation_via_membranes.bubble_pivoting.data_collection.bubble_pivoting_env import BubblePivotingEnv
from manipulation_via_membranes.bubble_pivoting.data_collection.pivoting_env_data_collector import PivotingEnvDataCollector

# TEST THE CODE: ------------------------------------------------------------------------------------------------------

def collect_data_pivoting_env(save_path, scene_name, force_threshold, max_force, grasp_width_limits, 
                              delta_y_limits, delta_roll_limits, roll_limits, tool,
                              num_data=10, impedance_mode=False, reactive=False):

    env = BubblePivotingEnv(impedance_mode=impedance_mode,
                            reactive=reactive,
                            force_threshold=force_threshold,
                            max_force=max_force,
                            grasp_width_limits=grasp_width_limits,
                            delta_y_limits=delta_y_limits,
                            delta_roll_limits=delta_roll_limits,
                            roll_limits=roll_limits,
                            tool=tool,
                            wrap_data=True
                           )
    dc = PivotingEnvDataCollector(env, data_path=save_path, scene_name=scene_name)
    dc.collect_data(num_data=num_data)



if __name__ == '__main__':

    parser = argparse.ArgumentParser('Collect Data Pivoting')
    parser.add_argument('save_path', type=str, help='path to save the data')
    parser.add_argument('num_data', type=int, help='Number of data samples to be collected')
    parser.add_argument('--scene_name', type=str, default='drawing_data', help='scene name for the data. For organization purposes')
    parser.add_argument('--force_threshold', type=float, default=10., help='force threshold to determine contact when lowering tool')
    parser.add_argument('--max_force', type=float, default=15., help='force threshold to determine dangerous interction with the environment')
    parser.add_argument('--impedance', action='store_true', help='impedance mode')
    parser.add_argument('--reactive', action='store_true', help='reactive mode -- adjust tool position to be straight when we start drawing')
    parser.add_argument('--grasp_width_limits', type=float, nargs=2, default=(5.0, 40.0), help='limits of the grasping force action space')
    parser.add_argument('--delta_y_limits', type=float, nargs=2, default=(-.04, .04), help='limits of the y movement during pivoting action')
    parser.add_argument('--delta_roll_limits', type=float, nargs=2, default=(-np.pi/6, np.pi/6), help='limits of the rotation during pivoting action')
    parser.add_argument('--tool', type=str, default='medium_stick', help='Tool code')
    parser.add_argument('--roll_limits', type=float, nargs=2, default=(np.pi/2,3*np.pi/2), help='limits of initial gripper rotation')

    args = parser.parse_args()

    save_path = args.save_path
    scene_name = args.scene_name
    num_data = args.num_data
    impedance_mode = args.impedance
    reactive = args.reactive
    force_threshold = args.force_threshold
    max_force = args.max_force
    grasp_width_limits = args.grasp_width_limits
    delta_y_limits = args.delta_y_limits
    delta_roll_limits = args.delta_roll_limits
    tool = args.tool
    roll_limits = args.roll_limits

    try:
        rospy.init_node('pivoting_data_collection')
    except (rospy.exceptions.ROSInitException, rospy.exceptions.ROSException):
        pass
    
    collect_data_pivoting_env(save_path, scene_name, force_threshold, max_force, grasp_width_limits, 
                              delta_y_limits, delta_roll_limits, roll_limits, tool,
                              num_data=num_data, impedance_mode=impedance_mode, reactive=reactive)
    # collect_data_pivoting(save_path, scene_name, num_data=num_data)