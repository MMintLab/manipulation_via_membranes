#!/usr/bin/env python3

import sys
import os
import rospy
import numpy as np
import argparse
import copy
import threading

from bubble_drawing.bubble_pose_estimation.bubble_pose_estimation import BubblePoseEstimator
from bubble_drawing.aux.load_confs import load_bubble_reconstruction_params
from bubble_drawing.bubble_contact_point_estimation.tool_contact_point_estimator import ToolContactPointEstimator


def estimate_contact_point():
    tcpe = ToolContactPointEstimator()

if __name__ == '__main__':
    # load params:
    params = load_bubble_reconstruction_params()
    object_names = list(params.keys())
    parser = argparse.ArgumentParser('Bubble Object Pose Estimation')
    parser.add_argument('object_name', type=str, help='Name of the object. Possible values: {}'.format(object_names))
    parser.add_argument('--reconstruction', type=str, default='tree', help='Name of imprint extraction algorithm. Possible values: (tree, depth)')
    parser.add_argument('--estimation_type', type=str, default='icp2d', help='Name of the algorithm used to estimate the pose from the imprint pc. Possible values: (icp2d, icp3d)')
    parser.add_argument('--rate', type=float, default=5.0, help='Estimated pose publishing rate (upper bound)')
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--imprint_br', action='store_true')
    parser.add_argument('--percentile', type=float, default=None, help='Percentile used for imprint filtering')

    args = parser.parse_args()

    object_name = args.object_name
    object_params = params[object_name]
    imprint_th = object_params['imprint_th'][args.reconstruction]
    icp_th = object_params['icp_th']
    gripper_width = object_params['gripper_width']

    print('-- Estimating the pose of a {} --'.format(object_name))


    bpe = BubblePoseEstimator(object_name=object_name,
                              imprint_th=imprint_th,
                              icp_th=icp_th,
                              rate=args.rate,
                              view=args.view,
                              percentile=args.percentile,
                              verbose=args.verbose,
                              broadcast_imprint=args.imprint_br,
                              estimation_type=args.estimation_type,
                              reconstruction=args.reconstruction,
                              gripper_width=gripper_width)







