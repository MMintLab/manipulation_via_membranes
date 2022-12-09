#! /usr/bin/env python
import numpy as np
import rospy
import tf
import tf.transformations as tr

from arm_robots.med import Med
from arc_utilities.listener import Listener
import tf2_geometry_msgs  # Needed by TF2Wrapper
from arc_utilities.tf2wrapper import TF2Wrapper
from victor_hardware_interface.victor_utils import Stiffness
from victor_hardware_interface_msgs.msg import ControlMode
from bubble_utils.bubble_med.bubble_med import BubbleMed

from geometry_msgs.msg import WrenchStamped
from visualization_msgs.msg import Marker

from bubble_control.bubble_contact_point_estimation.contact_point_marker_publisher import ContactPointMarkerPublisher


class BubbleDrawer(BubbleMed):

    def __init__(self, *args, object_topic='estimated_object', drawing_frame='med_base', force_threshold=5., reactive=False, adjust_lift=False, compensate_xy_point=False, impedance_mode=True, **kwargs):
        self.object_topic = object_topic
        self.drawing_frame = drawing_frame
        self.reactive = reactive # adjust drawing at keypoints/
        self.adjust_lift = adjust_lift
        self.force_threshold = force_threshold
        self.impedance_mode = impedance_mode
        # Parameters:
        self.draw_contact_gap = 0.005  # TODO: Consider reducing this value to reduce the force
        self.pre_height = 0.13
        self.draw_height_limit = 0.075  # we could go as lower as 0.06
        self.draw_quat = np.array([-np.cos(np.pi / 4), np.cos(np.pi / 4), 0, 0])
        self.marker_pose = None
        self.calibration_wrench = None
        self.compensate_xy_point = compensate_xy_point
        super().__init__(*args, **kwargs)
        self.pose_listener = Listener(self.object_topic, Marker, wait_for_data=False)
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.contact_point_marker_publisher = ContactPointMarkerPublisher()
        self.setup()

    def set_grasp_pose(self):
        self.set_robot_conf('grasp_conf')

    def setup(self):
        self.home_robot()
        self.calibration_wrench = self.get_wrench()

    def _stop_signal(self, feedback):
        wrench_stamped = self.get_wrench()
        measured_fz = wrench_stamped.wrench.force.z
        calibrated_fz = measured_fz-self.calibration_wrench.wrench.force.z
        flag_force = np.abs(calibrated_fz) >= np.abs(self.force_threshold)
        if flag_force:
            # print('force z: {} (measured: {}) --- flag: {} ({} | {})'.format(calibrated_fz, measured_fz, flag_force, np.abs(calibrated_fz), np.abs(self.force_threshold)))
            self.contact_point_marker_publisher.show = True
        return flag_force

    def _set_vel(self, vel):
        if self.impedance_mode:
            self.set_control_mode(ControlMode.JOINT_IMPEDANCE, stiffness=Stiffness.STIFF,vel=vel)
        else:
            self.set_control_mode(ControlMode.JOINT_POSITION, vel=vel)

    def get_plane_pose(self, child_frame='grasp_frame'):
        plane_pose_matrix = self.tf2_listener.get_transform(parent=self.drawing_frame, child=child_frame)
        plane_pose = self._matrix_to_pose(plane_pose_matrix)
        return plane_pose

    def get_marker_pose(self):
        data = self.pose_listener.get(block_until_data=True)
        pose = [data.pose.position.x,
                data.pose.position.y,
                data.pose.position.z,
                data.pose.orientation.x,
                data.pose.orientation.y,
                data.pose.orientation.z,
                data.pose.orientation.w
                ]
        marker_pose = {
            'pose': pose,
            'frame': data.header.frame_id,
        }
        return marker_pose

    def get_tool_angle_axis(self, ref_frame=None):
        if ref_frame is None:
            ref_frame = self.drawing_frame
        tool_frame_rf = self.tf2_listener.get_transform(parent=ref_frame, child='tool_frame')
        z_axis = np.array([0, 0, 1])
        tool_axis_tf = np.array([0,0,-1]) # Tool axis on the tool_frame
        tool_axis_tf_h = np.append(tool_axis_tf,0)
        tool_axis_rf_h = tool_frame_rf @ tool_axis_tf_h
        tool_axis_rf = tool_axis_rf_h[:-1] # remove homogeneous vector coordinates
        tool_angle = np.arccos(np.dot(tool_axis_rf, z_axis))
        _rot_axis = np.cross(tool_axis_rf, z_axis)
        rot_axis = _rot_axis / np.linalg.norm(_rot_axis)
        # TODO: account for tool_axis parallel to z_axis
        return tool_angle, rot_axis

    def get_contact_point(self, ref_frame=None):
        if ref_frame is None:
            ref_frame = self.drawing_frame
        contact_point_tf = self.tf2_listener.get_transform(parent=ref_frame, child='tool_contact_point')
        contact_xyz = contact_point_tf[:3,3]
        return contact_xyz

    def lower_down(self, z_value=None):
        if z_value is None:
            z_value = self.draw_height_limit
        # TODO: Consider entering on impedance mode
        self.force_threshold = 5.
        self.calibration_wrench = self.get_wrench()
        self.set_xyz_cartesian(z_value=z_value, frame_id='grasp_frame', ref_frame=self.drawing_frame,
                                   stop_condition=self._stop_signal)
        rospy.sleep(.5)
        # Read the z value achieved
        contact_pose = self.tf2_listener.get_transform(self.drawing_frame, 'grasp_frame')
        contact_z = contact_pose[2, 3]
        return contact_z

    def raise_up(self, z_value=None):
        if z_value is None:
            z_value = self.pre_height
        self.set_xyz_cartesian(z_value=z_value, frame_id='grasp_frame', ref_frame=self.drawing_frame)
        self.contact_point_marker_publisher.show = False  # deactivate the force flag # TODO: Make this a function of the force

    def compensate_tool_position(self):
        contact_point = self.get_contact_point()
        angle, axis = self.get_tool_angle_axis()

        plan = self.rotation_along_axis_point_angle(axis=axis, angle=angle, point=contact_point,
                                             num_steps=20, position_tol=0.001, orientation_tol=0.005) # TODO: Set as hyperparameter
        return plan

    def _get_marker_compensated_pose(self, desired_marker_pose, ref_frame=None, tf_broadcast=False):
        if ref_frame is None:
            ref_frame = self.drawing_frame
        desired_position, desired_quat = np.split(desired_marker_pose, [3])
        T_desired = tr.quaternion_matrix(desired_quat)  # in world frame
        T_desired[:3, 3] = desired_position
        current_marker_pose = self.get_marker_pose()
        T_mf = tr.quaternion_matrix(current_marker_pose['pose'][3:])  # marker frame in grasp frame
        T_mf[:3, 3] = current_marker_pose['pose'][:3]
        T_mf_desired = T_desired @ np.linalg.inv(T_mf)
        target_pose = np.concatenate([T_mf_desired[:3, 3], tr.quaternion_from_matrix(T_mf_desired)]) # pose of the frame wrt the marker pose is broadcasted, a.k.a current_marker_pose['frame']
        target_pose_frame = current_marker_pose['frame']
        if tf_broadcast:
            # update the tfs for the frames
            self.tf_broadcaster.sendTransform(list(target_pose[:3]), list(target_pose[3:]), rospy.Time.now(),
                                              '{}_desired'.format(current_marker_pose['frame']), ref_frame)

            self.tf_broadcaster.sendTransform(list(desired_marker_pose[:3]), list(desired_marker_pose[3:]), rospy.Time.now(),
                                              'desired_obj_pose', ref_frame)
            self.tf_broadcaster.sendTransform(list(current_marker_pose['pose'][:3]),
                                              list(current_marker_pose['pose'][3:]), rospy.Time.now(),
                                              'current_obj_pose', current_marker_pose['frame'])
        return {'pose': target_pose, 'frame': target_pose_frame}

    def _init_drawing(self, init_point_xy, draw_quat=None, ref_frame=None):
        # Plan to the point_xy and perform a guarded move to start drawing
        # Variables:
        pre_height = self.pre_height
        draw_height_limit = self.draw_height_limit
        draw_z = draw_height_limit
        draw_z = 0.065
        draw_contact_gap = 0.005  # TODO: Consider reducing this value to reduce the force

        if draw_quat is None:
            draw_quat = self.draw_quat

        if ref_frame is None:
            ref_frame = self.drawing_frame

        # first plan to the first corner
        pre_position = np.insert(init_point_xy, 2, pre_height)
        pre_pose = np.concatenate([pre_position, draw_quat], axis=0)

        if self.reactive:
            # Account for the pose of the marker in-hand.
            target_pose_dict = self._get_marker_compensated_pose(desired_marker_pose=pre_pose, ref_frame=ref_frame, tf_broadcast=True)
            self.plan_to_pose(self.arm_group, target_pose_dict['frame'], target_pose=list(target_pose_dict['pose']),
                                  frame_id=self.drawing_frame)
        else:
            self.plan_to_pose(self.arm_group, 'grasp_frame', target_pose=list(pre_pose), frame_id=self.drawing_frame)
        # self.med.set_control_mode(ControlMode.JOINT_IMPEDANCE, stiffness=Stiffness.STIFF, vel=0.075)  # Low val for safety
        rospy.sleep(.5)
        self._set_vel(0.03)  # Very Low val for precision
        # lower down:
        self.force_threshold = 5.
        contact_z = self.lower_down(z_value=draw_z)
        # Read the value of z when we make contact to only set a slight force on the plane
        draw_height = max(contact_z - draw_contact_gap, draw_height_limit)
        # read force
        first_contact_wrench = self.get_wrench()
        # print('contact wrench: ', first_contact_wrench.wrench)
        self.force_threshold = 18  # Increase the force threshold

        self._set_vel(0.1)  # change speed
        return draw_height

    def _draw_to_point(self, point_xy, draw_height, end_raise=False):
        if self.compensate_xy_point:
            tcp_gf = self.tf_wrapper.get_transform(
                'tool_contact_point',
                'grasp_frame')  # transform from the contact point to the grasp frame
            point_xy = point_xy + tcp_gf[:2,3]
        position_i = np.insert(point_xy, 2, draw_height) # position of the poin in the world frame

        plan_result = self.plan_to_position_cartesian(self.arm_group, 'grasp_frame', target_position=list(position_i),stop_condition=self._stop_signal)
        if end_raise:
            # Raise the arm when we reach the last point
            self._end_raise(point_xy)
        return plan_result

    def _end_raise(self, point_xy=None):
        if point_xy is not None:
            # set the raise on the xy
            final_position = np.insert(point_xy, 2, self.pre_height)
            final_pose = np.concatenate([final_position, self.draw_quat], axis=0)
            self.plan_to_pose(self.arm_group, 'grasp_frame', target_pose=list(final_pose), frame_id=self.drawing_frame)
        else:
            # just raise up on z direction
            self.set_xyz_cartesian(z_value=self.pre_height)

    def _adjust_tool_position(self, xy_point, lift=None):
        if lift is None:
            lift = self.adjust_lift
        # Adjust the position when we reach the keypoint -------
        # Lift:
        # draw_quat = self.draw_quat
        # lift_position = np.insert(xy_point, 2, self.pre_height)
        # self.plan_to_position_cartesian(self.arm_group, 'grasp_frame', target_position=list(lift_position))
        if lift:
            self.raise_up()

            # compensate for the orientation of the marker
            desired_marker_pos = np.insert(xy_point, 2, self.pre_height)
            desired_marker_pose = np.concatenate([desired_marker_pos, self.draw_quat])
            compensated_marker_pose = self._get_marker_compensated_pose(desired_marker_pose)
            plan_result = self.plan_to_pose(self.arm_group, compensated_marker_pose['frame'],
                                                target_pose=list(compensated_marker_pose['pose']), frame_id=self.drawing_frame)
            # Go down again
            rospy.sleep(.5)
            self._set_vel(0.05)  # Very Low val for precision

            # lower down:
            contact_z = self.lower_down()
            draw_height = max(contact_z - self.draw_contact_gap, self.draw_height_limit)
            # read force
            first_contact_wrench = self.get_wrench()
            # print('contact wrench: ', first_contact_wrench.wrench)
            self.force_threshold = 18  # Increase the force threshold

            self._set_vel(0.1)
            rospy.sleep(.5)
        else:
            # Rotate along the contact point to correct the tool positions
            self.compensate_tool_position()
            current_pose = self.get_current_pose()
            # draw_height = self.draw_height_limit
            draw_height = max(current_pose[2]-0.001, self.draw_height_limit)
        return draw_height

    def draw_points(self, xy_points, end_raise=True, end_adjust=True, init_drawing=True):
        """
        Draw lines between a series of xy points. The robot executes cartesian trajectories on impedance mode between all points on the list
        Args:
            xy_points: <np.ndarray> of size (N,2) containing the N points on the xy plane we want to be drawn

        Returns: None
        """
        # TODO: Split into guarded moved motion and drawing_motion between points
        if init_drawing:
            draw_height = self._init_drawing(init_point_xy=xy_points[0])
        else:
            contact_pose = self.tf2_listener.get_transform(self.drawing_frame, 'grasp_frame')
            draw_height = contact_pose[2, 3]
        rospy.sleep(.5)
        self._set_vel(0.1)
        for i, corner_i in enumerate(xy_points[1:]):
            self._draw_to_point(corner_i, draw_height, end_raise=False)
            if self.reactive and (i < len(xy_points)-2):
                # Adjust the position when we reach the keypoint -------
                draw_height = self._adjust_tool_position(corner_i, lift=self.adjust_lift) # TODO: Consider using only the initial draw_height and do not update it every adjustment
        if end_adjust and not self.adjust_lift:
            self.compensate_tool_position()
        if end_raise:
            # Raise the arm when we reach the last point
            self.raise_up()

    def draw_square(self, side_size=0.2, center=(0.55, -0.1), step_size=None, spread_evenly=True, **kwargs):
        corners = np.asarray(center) + side_size * 0.5 * np.array([[1, 1], [1, -1], [-1, -1], [-1, 1], [1,1]])
        if step_size is not None:
            corners = self._discretize_points(corners, step_size=step_size, spread_evenly=spread_evenly)
        self.draw_points(corners, **kwargs)

    def draw_regular_polygon(self, num_sides, circumscribed_radius=0.2, center=(0.55, -0.1), init_angle=0, step_size=None, **kwargs):
        _angles = 2 * np.pi * np.arange(num_sides+1)/(num_sides)
        angles = (init_angle + _angles )%(2*np.pi)
        basic_vertices = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        corners = np.asarray(center) + circumscribed_radius * 0.5 * basic_vertices
        if step_size is not None:
            corners = self._discretize_points(corners, step_size=step_size)
        self.draw_points(corners, **kwargs)

    def draw_circle(self, radius=0.2, num_points=100, center=(0.55, -0.1), **kwargs):
        self.draw_regular_polygon(num_sides=num_points, circumscribed_radius=radius, center=center, **kwargs)

    def _discretize_points(self, points_xy, step_size=0.05, spread_evenly=True):
        """
        Given a set of points on xy plane, create a new set of points where points at most are step_size from each other
        Args:
            points_xy:
        Returns:
        """
        num_keypoints = points_xy.shape[0]
        point_dim = points_xy.shape[-1]
        discretized_points = []
        for i, point_i in enumerate(points_xy[:-1]):
            next_point = points_xy[i+1]
            delta_v = next_point - point_i
            point_dist = np.linalg.norm(delta_v)
            unit_v = delta_v/point_dist
            num_points = int(point_dist//step_size)
            if spread_evenly:
                # spread the points so they are all evenly distributed
                step_i = point_dist/num_points
            else:
                # points dist a fixed distance, except last one that has a residual distance <= step_size
                step_i = step_size
            points_i = point_i + step_i * np.stack([np.arange(num_points)]*point_dim, axis=-1) * np.stack([unit_v]*num_points)
            for new_point in points_i:
                discretized_points.append(new_point)
        discretized_points.append(points_xy[-1])
        discretized_points = np.stack(discretized_points)
        return discretized_points

    def control(self, desired_pose, ref_frame):
        """
        Adjust the robot so the object has a constant pose (target_pose) in the reference ref_frame
        Args:
            target_pose: <list> pose as [x,y,z,qw,qx,qy,qz]
            ref_frame: <str>
        """
        T_desired = tr.quaternion_matrix(desired_pose[3:])
        T_desired[:3, 3] = desired_pose[:3]
        try:
            while not rospy.is_shutdown():
                print('Control')
                # Read object position
                current_marker_pose = self.get_marker_pose()
                T_mf = tr.quaternion_matrix(current_marker_pose['pose'][3:])
                T_mf[:3,3] = current_marker_pose['pose'][:3]
                T_mf_desired = T_desired @ np.linalg.inv(T_mf)   # maybe it is this

                # Compute the target
                target_pose = np.concatenate([T_mf_desired[:3,3], tr.quaternion_from_matrix(T_mf_desired)])
                # broadcast target_pose:
                self.tf_broadcaster.sendTransform(list(target_pose[:3]), list(target_pose[3:]), rospy.Time.now(), '{}_desired'.format(current_marker_pose['frame']), ref_frame)
                self.tf_broadcaster.sendTransform(list(desired_pose[:3]), list(desired_pose[3:]), rospy.Time.now(), 'desired_obj_pose', ref_frame)
                self.tf_broadcaster.sendTransform(list(current_marker_pose['pose'][:3]), list(current_marker_pose['pose'][3:]), rospy.Time.now(), 'current_obj_pose', current_marker_pose['frame'])
                plan_result = self.plan_to_pose(self.arm_group, current_marker_pose['frame'], target_pose=list(target_pose), frame_id=ref_frame)
        except rospy.ROSInterruptException:
            pass

    def test_pivot_motion(self):
        world_frame_name = self.drawing_frame
        contact_point_frame_name = 'tool_contact_point'
        tool_frame_name = 'tool_frame'
        tool_frame_desired_quat = np.array([-np.cos(np.pi/4), np.cos(np.pi/4), 0, 0]) # desired quaternion in the world frame (med_base)
        contact_frame_wf = self.tf2_listener.get_transform(parent=world_frame_name, child=contact_point_frame_name)
        tool_frame_cpf = self.tf2_listener.get_transform(parent=contact_point_frame_name, child=tool_frame_name) # tool frame on the contact point frame
        tool_frame_gf = self.tf2_listener.get_transform(parent='grasp_frame', child=tool_frame_name) # tool frame on grasp frame

        tool_frame_t_cpf = tool_frame_cpf[:3, 3]
        final_axis_t_cpf = np.array([0, 0, 1])
        _desired_tool_cf = tr.quaternion_matrix(tool_frame_desired_quat)
        _desired_tool_cf[:3, 3] = final_axis_t_cpf * np.linalg.norm(tool_frame_t_cpf)
        desired_tool_cf = _desired_tool_cf
        desired_tool_wf = contact_frame_wf @ desired_tool_cf # w_T_cf @ cf_T_dtf = w_T_dtf
        desired_grasp_frame_wf = desired_tool_wf @ tr.inverse_matrix(tool_frame_gf)# w_T_gf = w_T_dtf @ dtf_T_gf
        desired_grasp_pose = list(desired_grasp_frame_wf[:3, 3]) + list(tr.quaternion_from_matrix(desired_grasp_frame_wf))
        desired_tool_pose = list(desired_tool_wf[:3,3]) + list(tr.quaternion_from_matrix(desired_tool_wf))
        # try all at one:
        self.tf2_listener.send_transform(translation=desired_grasp_pose[:3], quaternion=desired_grasp_pose[3:], parent=world_frame_name, child='desired_grasp_frame_pivot', is_static=True)
        self.tf2_listener.send_transform(translation=desired_tool_pose[:3], quaternion=desired_tool_pose[3:], parent=world_frame_name, child='desired_tool_frame_pivot', is_static=True)
        self.plan_to_pose(self.arm_group, ee_link_name='grasp_frame', target_pose=desired_grasp_pose, frame_id=world_frame_name, stop_condition=self._stop_signal)


# TEST THE CODE: ------------------------------------------------------------------------------------------------------

def draw_test(supervision=False, reactive=False):
    bd = BubbleDrawer(reactive=reactive)
    # center = (0.55, -0.25) # good one
    center = (0.45, -0.25)
    # center_2 = (0.55, 0.2)
    center_2 = (0.45, 0.2)

    # bd.draw_square()
    # bd.draw_regular_polygon(3, center=center)
    # bd.draw_regular_polygon(4, center=center)
    # bd.draw_regular_polygon(5, center=center)
    # bd.draw_regular_polygon(6, center=center)
    for i in range(5):
        # bd.draw_square(center=center_2)
        bd.draw_regular_polygon(3, center=center, circumscribed_radius=0.15)
    # bd.draw_square(center=center, step_size=0.04)


    # bd.draw_square(center=center_2)
    # bd.draw_square(center=center_2)
    # bd.draw_square(center=center_2)
    # bd.draw_square(center=center_2, step_size=0.04)

    # bd.draw_circle()

def reactive_demo():
    bd = BubbleDrawer(reactive=True)
    center = (0.6, -0.25)
    center_2 = (0.6, 0.2)

    num_iters = 5

    for i in range(num_iters):
        bd.draw_regular_polygon(4, center=center, circumscribed_radius=0.15, init_angle=np.pi*0.25)

    _ = input('Please, rearange the marker and press enter. ')
    bd.reactive = False
    for i in range(num_iters):
        bd.draw_regular_polygon(4, center=center_2, circumscribed_radius=0.15, init_angle=np.pi*0.25)

def test_pivot():
    bd = BubbleDrawer(reactive=True)
    while True:
        _ = input('press enter to continue')
        bd.test_pivot_motion()

def draw_M():
    bd = BubbleDrawer(reactive=False, force_threshold=0.25)
    m_points = np.load('/home/mmint/InstalledProjects/robot_stack/src/bubble_control/config/M.npy')
    m_points[:,1] = m_points[:,1]*(-1)
    # scale points:
    scale = 0.25
    corner_point = np.array([.75, .1])
    R = tr.quaternion_matrix(tr.quaternion_about_axis(angle=-np.pi*0.5, axis=np.array([0, 0, 1])))[:2, :2]
    m_points_rotated = m_points @ R.T
    m_points_scaled = corner_point + scale*m_points_rotated
    m_points_scaled = np.concatenate([m_points_scaled, m_points_scaled[0:1]], axis=0)
    bd.draw_points(m_points_scaled)

if __name__ == '__main__':
    supervision = False
    reactive = True
    # reactive = False

    # topic_recorder = TopicRecorder()
    # wrench_recorder = WrenchRecorder('/med/wrench', ref_frame='world')
    # wrench_recorder.record()
    # draw_test(supervision=supervision, reactive=reactive)
    # print('drawing done')
    # wrench_recorder.stop()
    # wrench_recorder.save('~/Desktop')

    # reactive_demo()
    # test_pivot()
    draw_M()