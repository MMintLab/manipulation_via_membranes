#! /usr/bin/env python
import os
import pdb
import sys
import numpy as np
import threading
import copy
import rospy
import torch
import tf
import tf.transformations as tr
import argparse

from bubble_control.bubble_data_collection.bubble_draw_data_collection import BubbleDrawingDataCollection
from bubble_control.bubble_envs.bubble_drawing_env import BubbleCartesianDrawingEnv, BubbleOneDirectionDrawingEnv, BubbleLineDrawingEnv
from bubble_utils.bubble_data_collection.env_data_collection import EnvDataCollector, ReferencedEnvDataCollector
from bubble_control.bubble_model_control.aux.bubble_dynamics_fixed_model import BubbleDynamicsFixedModel
from bubble_control.bubble_model_control.aux.format_observation import format_observation_sample
from bubble_control.bubble_model_control.cost_functions import vertical_tool_cost_function
from bubble_control.bubble_model_control.controllers.bubble_model_mppi_controler import BubbleModelMPPIController
from bubble_control.bubble_model_control.model_output_object_pose_estimaton import BatchedModelOutputObjectPoseEstimation
from bubble_control.bubble_model_control.drawing_action_models import drawing_action_model_one_dir
from bubble_control.bubble_envs.controlled_env import ControlledEnvWrapper
from bubble_control.bubble_model_control.aux.format_observation import format_observation_sample
from bubble_control.bubble_learning.aux.img_trs.block_downsampling_tr import BlockDownSamplingTr

# TEST THE CODE: ------------------------------------------------------------------------------------------------------


def collect_data_drawing_test(save_path, scene_name, num_data=10, prob_axis=0.08, impedance_mode=False, reactive=False, drawing_area_center=(0.55, 0.), drawing_area_size=(.15, .15), drawing_length_limits=(0.01, 0.15)):

    dc = BubbleDrawingDataCollection(data_path=save_path,
                                     scene_name=scene_name,
                                     prob_axis=prob_axis,
                                     impedance_mode=impedance_mode,
                                     reactive=reactive,
                                     drawing_area_center=drawing_area_center,
                                     drawing_area_size=drawing_area_size,
                                     drawing_length_limits=drawing_length_limits)
    dc.collect_data(num_data=num_data)


def collect_data_drawing_env_test(save_path, scene_name, num_data=10, prob_axis=0.08, impedance_mode=False, reactive=False, drawing_area_center=(0.55, 0.), drawing_area_size=(.15, .15), drawing_length_limits=(0.01, 0.15), grasp_width_limits=(15, 25), object_name='marker'):

    env = BubbleOneDirectionDrawingEnv(prob_axis=prob_axis,
                             impedance_mode=impedance_mode,
                             reactive=reactive,
                             drawing_area_center=drawing_area_center,
                             drawing_area_size=drawing_area_size,
                             drawing_length_limits=drawing_length_limits,
                             grasp_width_limits=grasp_width_limits,
                             wrap_data=True,
                                       marker_code=object_name,
                           )
    dc = ReferencedEnvDataCollector(env, data_path=save_path, scene_name=scene_name)
    dc.collect_data(num_data=num_data)


def collect_data_drawing_env_jacobian_controller(save_path, scene_name, num_data=10, prob_axis=0.08, impedance_mode=False,
                                                 reactive=False, drawing_area_center=(0.55, 0.), drawing_area_size=(.15, .15),
                                                 drawing_length_limits=(0.01, 0.15), grasp_width_limits=(10, 40),
                                                 object_name='marker', num_samples=100, horizon=2, random_action_prob=0.15,
                                                 ):

    # env = BubbleOneDirectionDrawingEnv(
    env = BubbleLineDrawingEnv(
             prob_axis=prob_axis,
             impedance_mode=impedance_mode,
             reactive=reactive,
             drawing_area_center=drawing_area_center,
             drawing_area_size=drawing_area_size,
             drawing_length_limits=drawing_length_limits,
             grasp_width_limits=grasp_width_limits,
             marker_code=object_name,
             wrap_data=True
                           )
    model = BubbleDynamicsFixedModel() # Fixed model for Jacobian controller
    ope = BatchedModelOutputObjectPoseEstimation(object_name='marker', factor_x=7, factor_y=7, method='bilinear',
                                                 device=torch.device('cuda'), imprint_selection='percentile',
                                                 imprint_percentile=0.005)

    block_downsample_tr = BlockDownSamplingTr(factor_x=7, factor_y=7, reduction='mean', keys_to_tr=['init_imprint'])

    controller = BubbleModelMPPIController(model, env, ope, vertical_tool_cost_function,
                                                  action_model=drawing_action_model_one_dir, num_samples=num_samples,
                                                  horizon=horizon, noise_sigma=None, _noise_sigma_value=.3, state_trs=(format_observation_sample, block_downsample_tr))


    controlled_env = ControlledEnvWrapper(env=env, controller=controller, random_action_prob=random_action_prob, controlled_action_keys=['rotation', 'length'])
    dc = ReferencedEnvDataCollector(controlled_env, data_path=save_path, scene_name=scene_name)
    dc.collect_data(num_data=num_data)



if __name__ == '__main__':

    parser = argparse.ArgumentParser('Collect Data Drawing')
    parser.add_argument('save_path', type=str, help='path to save the data')
    parser.add_argument('num_data', type=int, help='Number of data samples to be collected')
    parser.add_argument('--scene_name', type=str, default='drawing_data', help='scene name for the data. For organization purposes')
    parser.add_argument('--prob_axis', type=float, default=0.08, help='probability for biasing the drawing along the axis')
    parser.add_argument('--impedance', action='store_true', help='impedance mode')
    parser.add_argument('--reactive', action='store_true', help='reactive mode -- adjust tool position to be straight when we start drawing')
    parser.add_argument('--drawing_area_center', type=float, nargs=2, default=(0.55, 0.), help='x y of the drawing area center')
    parser.add_argument('--drawing_area_size', type=float, nargs=2, default=(0.15, 0.3), help='delta_x delta_y of the semiaxis drawing area')
    parser.add_argument('--drawing_length_limits', type=float, nargs=2, default=(0.01, 0.02), help='min_length max_length of the drawing move')
    parser.add_argument('--grasp_width_limits', type=float, nargs=2, default=(10, 35), help='min and max grasp width values')
    parser.add_argument('--controlled', action='store_true', help='collect data using a controlled policy -- with probability random_action_prob perform a random action')
    parser.add_argument('--random_action_prob', type=float,  default=0.15, help='probability of performing random actions')
    parser.add_argument('--object_name', type=str,  default='marker', help='name of the grasped object, we may record it on our dataset')

    args = parser.parse_args()

    save_path = args.save_path
    scene_name = args.scene_name
    num_data = args.num_data
    prob_axis = args.prob_axis
    impedance_mode = args.impedance
    reactive = args.reactive
    drawing_area_center = args.drawing_area_center
    drawing_area_size = args.drawing_area_size
    drawing_length_limits = args.drawing_length_limits
    grasp_width_limits = args.grasp_width_limits
    random_action_prob = args.random_action_prob
    controlled = args.controlled
    object_name = args.object_name

    if controlled:
        collect_data_drawing_env_jacobian_controller(save_path, scene_name, num_data=num_data, prob_axis=prob_axis,
                                      impedance_mode=impedance_mode, reactive=reactive,
                                      drawing_area_center=drawing_area_center,
                                      drawing_area_size=drawing_area_size, drawing_length_limits=drawing_length_limits,
                                      grasp_width_limits=grasp_width_limits, random_action_prob=random_action_prob, object_name=object_name)
    else:
        collect_data_drawing_env_test(save_path, scene_name, num_data=num_data, prob_axis=prob_axis,
                              impedance_mode=impedance_mode, reactive=reactive, drawing_area_center=drawing_area_center,
                              drawing_area_size=drawing_area_size, drawing_length_limits=drawing_length_limits, grasp_width_limits=grasp_width_limits, object_name=object_name)