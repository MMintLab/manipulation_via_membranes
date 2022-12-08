import abc
import numpy as np
from collections import OrderedDict
import gym
import copy
import tf.transformations as tr
from manipulation_via_membranes.bubble_pivoting.pivoting_model_control.aux.pivoting_geometry import get_angle_difference, get_tool_axis


class AxisBiasedDirectionSpace(gym.spaces.Space):
    """
    Saple space between [0,2pi) with bias towards the axis directions.
    On prob_axis, the sample will be along one of the cartesian axis directions, i.e. [0, pi/2, pi, 3pi/2]
    """
    def __init__(self, prob_axis, seed=None):
        """
        Args:
            prob_axis: probability of sampling a direction along the axis
            seed:
        """
        self.prob_axis = prob_axis
        super().__init__((), np.float32, seed)

    def sample(self):
        p_axis_direction = self.np_random.random() # probability of getting an axis motion
        if p_axis_direction < self.prob_axis:
            direction_i = 0.5 * np.pi * np.random.randint(0, 4) # axis direction (0, pi/2, pi/ 3pi/2)
        else:
            direction_i = np.random.uniform(0, 2 * np.pi)  # direction as [0, 2pi)
        return direction_i

    def contains(self, x):
        return 0 <= x <= 2*np.pi


class FinalPivotingPoseSpace(gym.spaces.Space):
    """
    Sample pivoting pose (with orientation as euler)
    """
    def __init__(self, med, current_pose, delta_y_limits, delta_z_limits, delta_roll_limits, seed=None):
        super().__init__((), np.float32, seed)
        self.current_pose = current_pose
        self.current_pose = np.concatenate((current_pose[:3], tr.euler_from_quaternion(current_pose[3:], 'sxyz')))
        self.delta_y_limits = delta_y_limits
        self.delta_z_limits = delta_z_limits
        self.delta_roll_limits = delta_roll_limits
        self.med = med
        self.low = np.array([self.current_pose[0], self.current_pose[1]+self.delta_y_limits[0], self.current_pose[2]+self.delta_z_limits[0],
                                self.current_pose[3] + self.delta_roll_limits[0], 0, np.pi])
        self.high = np.array([self.current_pose[0], self.current_pose[1]+self.delta_y_limits[1], self.current_pose[2]+self.delta_z_limits[1],
                                self.current_pose[3] + self.delta_roll_limits[1], 0, np.pi])

    def sample(self):
        delta_y, delta_z = np.random.uniform(np.array([self.delta_y_limits[0], self.delta_z_limits[0]]), 
                                            np.array([self.delta_y_limits[1], self.delta_z_limits[0]]))
        movement_wf = delta_y * np.array([0,1,0]) + delta_z * np.array([0,0,1])
        delta_roll_wf = np.random.uniform(self.delta_roll_limits[0], self.delta_roll_limits[1])
        pose_w_quat = np.concatenate((self.current_pose[:3], tr.quaternion_from_euler(self.current_pose[3], self.current_pose[4], self.current_pose[5], 'sxyz')))
        orientation = self.med._compute_rotation_along_axis_point_angle(pose=pose_w_quat, 
                        angle=delta_roll_wf, point=self.current_pose[:3], axis=np.array([1,0,0]))[3:]
        final_pose_wf = np.concatenate([self.current_pose[:3]+movement_wf, tr.euler_from_quaternion(orientation, 'sxyz')])
        return final_pose_wf

    #TODO: Add orientation limits
    def contains(self, position):
        lower_bound = np.array([self.current_pose[0], self.current_pose[1]+self.delta_y_limits[0], self.current_pose[2]+self.delta_z_limits[0],
                                self.current_pose[3] + self.delta_roll_limits[0], 0, np.pi])
        upper_bound = np.array([self.current_pose[0], self.current_pose[1]+self.delta_y_limits[1], self.current_pose[2]+self.delta_z_limits[1],
                                self.current_pose[3] + self.delta_roll_limits[1], 0, np.pi])
        return lower_bound  <= position <= upper_bound

class InitialPivotingPoseSpace(gym.spaces.Space):
    """
    Sample initial pose for pivoting (with orientation as euler)
    """
    def __init__(self, init_x_limits, init_y_limits, init_z_limits, roll_limits, seed=None):
        super().__init__((), np.float32, seed)
        self.init_x_limits = init_x_limits
        self.init_y_limits = init_y_limits
        self.init_z_limits = init_z_limits
        self.roll_limits = roll_limits
        self.low = np.array([self.init_x_limits[0], self.init_y_limits[0], self.init_z_limits[0], self.roll_limits[0], 0, np.pi]) #self.roll_limits[0] points away from user
        self.high = np.array([self.init_x_limits[1], self.init_y_limits[1], self.init_z_limits[1], self.roll_limits[1], 0, np.pi])

    def sample(self):
        initial_position = np.random.uniform(np.array([self.init_x_limits[0], self.init_y_limits[0], self.init_z_limits[0]]),
                                             np.array([self.init_x_limits[1], self.init_y_limits[1], self.init_z_limits[1]]))
        roll = np.random.uniform(self.roll_limits[0],self.roll_limits[1])
        initial_orientation = np.array([roll, 0, np.pi])
        initial_quaternion = tr.quaternion_from_euler(initial_orientation[0], initial_orientation[1], initial_orientation[2], 'sxyz')
        initial_pose_wf = np.concatenate([initial_position, initial_quaternion])                                            
        return initial_pose_wf

    def contains(self, pose):
        lower_bound = np.array([self.init_x_limits[0], self.init_y_limits[0], self.init_z_limits[0], self.roll_limits[0], 0, np.pi])
        upper_bound = np.array([self.init_x_limits[1], self.init_y_limits[1], self.init_z_limits[1], self.roll_limits[1], 0, np.pi])
        return lower_bound  <= pose <= upper_bound
    

class DeltaRollSpace(gym.spaces.Space):
    """
    Sample initial pose for pivoting (with orientation as euler)
    """
    def __init__(self, delta_roll_limits, seed=None):
        super().__init__((), np.float32, seed)
        self.delta_roll_limits = delta_roll_limits
        self.low = delta_roll_limits[0] #self.roll_limits[0] points away from user
        self.high = delta_roll_limits[1]

    def sample(self):
        p_direction = self.np_random.random()
        normal_d = np.random.normal(loc=0.0, scale=self.delta_roll_limits[1]/4)
        if p_direction < 0.5:
            delta_roll = self.delta_roll_limits[1] - normal_d
        else:
            delta_roll = self.delta_roll_limits[0] + normal_d                                        
        return delta_roll

    def contains(self, pose):
        lower_bound = np.array([self.init_x_limits[0], self.init_y_limits[0], self.init_z_limits[0], self.roll_limits[0], 0, np.pi])
        upper_bound = np.array([self.init_x_limits[1], self.init_y_limits[1], self.init_z_limits[1], self.roll_limits[1], 0, np.pi])
        return lower_bound  <= pose <= upper_bound
    
class RollSpace(gym.spaces.Space):
    """
    Sample initial roll for pivoting
    """
    def __init__(self, seed=None):
        self.high = None
        self.low = None
        self.seed = seed

    
    def sample(self, direction=None):
        self.tool_axis_gf = get_tool_axis(ref_frame='grasp_frame')
        tool_axis_wf = get_tool_axis(ref_frame='med_base')
        if tool_axis_wf[2] > 0:
            self.tool_axis_gf *= -1
        self.tool_angle_gf = get_angle_difference(child_axis=self.tool_axis_gf, parent_axis=np.array([0, 0, 1]))
        if direction is None:
            direction = np.random.choice([-1,1], 1)
            self.high = np.pi + np.pi/4 - self.tool_angle_gf
            self.low = np.pi - np.pi/4 - self.tool_angle_gf
            roll = np.pi + direction * np.pi/4 - self.tool_angle_gf
        else:
            roll = np.pi + direction * np.pi/4 - self.tool_angle_gf
            self.high = roll
            self.low = roll
        return roll


class ConstantSpace(gym.spaces.Space):
    """
    Constant space. Only has one possible value. For convenience.
    """
    def __init__(self, value, seed=None):
        self.value = value
        super().__init__((), np.float32, seed)

    def sample(self):
        return self.value

    def contains(self, x):
        return x == self.value

    def __eq__(self, other):
        return (
                isinstance(other, ConstantSpace)
                and self.value == other.value
        )


class DiscreteElementSpace(gym.spaces.Space):
    """
    Space given by a discrete set of elements
    """
    def __init__(self, elements, probs=None, seed=None):
        self.elements = elements
        self.probs = probs
        if self.probs is None:
            self.probs = 1/self.num_elements*np.ones(self.num_elements)
        super().__init__((), np.float32, seed)

    @property
    def num_elements(self):
        return len(self.elements)

    def sample(self):
        element_sampled = np.random.choice(self.elements, p=self.probs)
        return element_sampled

    def contains(self, value):
        return value in self.elements

    def __eq__(self, other):
        return (
            isinstance(other, DiscreteElementSpace) and self.elements == other.elements
        )