import os
import pdb
import sys
import time
import numpy as np
import threading
import copy
import rospy
import tf
import tf.transformations as tr


class TFBufferPublisher(object):
    def __init__(self, rate=10.0):
        self.rate = rate
        self.is_alive = True
        self.lock = threading.Lock()
        self.tf = None
        self.timeout = None
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.publishing_thread = threading.Thread(target=self._publish_loop)

    def _publish_loop(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            with self.lock:
                if self.timeout is not None:
                    if self.timeout < time.time():
                        self.tf = None # Timeout reached, stop broadcasting
                tf = copy.deepcopy(self.tf)
            if tf is not None:
                # broadcast the tf
                self.tf_broadcaster.sendTransform(tf['pos'], tf['quat'], rospy.Time.now(), child=tf['child_frame_name'], parent=tf['parent_frame_name'])
            rate.sleep()
            with self.lock:
                if not self.is_alive:
                    return

    def send_tf(self, translation, quaternion, parent_frame_name, child_frame_name, timeout=None):
        """

        Args:
            translation:
            quaternion:
            parent_frame_name:
            child_frame_name:
            timeout: time to broadcast the tf in seconds. It will automatically stop after that time.
                If None, the broadcasting will keep going until finish() or reset() are called or another tf is passed to send_tf.
        """
        # TODO: Finish
        tf = {'pos': translation,
              'quat': quaternion,
              'parent_frame_name': parent_frame_name,
              'child_frame_name': child_frame_name}
        with self.lock:
            self.tf = tf
            if timeout is not None:
                self.timeout = time.time() + timeout

    def reset(self):
        with self.lock:
            self.tf = None
            self.timeout = None

    def finish(self):
        with self.lock:
            self.is_alive = False