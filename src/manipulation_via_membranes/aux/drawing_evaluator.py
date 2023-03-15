import rospy
import numpy as np
import cv2
import os

from arc_utilities.tf2wrapper import TF2Wrapper
from mmint_camera_utils.camera_utils.camera_parsers import RealSenseCameraParser
from mmint_tools.camera_tools.img_utils import project_points_pinhole
from matplotlib import pyplot as plt
from scipy.spatial import KDTree

from mmint_camera_utils.ros_utils.marker_publisher import MarkerPublisher
from geometry_msgs.msg import Point

def transform_points(points, X):
    points_original_shape = points.shape
    points = points.reshape(-1, points_original_shape[-1]) # add batch_dim
    points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
    points_tr_h = points_h @ X.T
    points_tr = points_tr_h[..., :3]
    points_tr = points_tr.reshape(points_original_shape)
    return points_tr


def transform_vectors(vectors, X):
    vectors_original_shape = vectors.shape
    vectors = vectors.reshape(-1, vectors_original_shape[-1]) # add batch_dim
    vectors_tr = vectors @ X[:3, :3].T
    vectors_tr = vectors_tr.reshape(vectors_original_shape)
    return vectors_tr


def invert_img(img):
    origin_type = img.dtype
    img = img.astype(np.float32)
    max_v = np.max(img)
    min_v = np.min(img)
    # set max_v to min_v and min_v to max_v
    img_norm = (img-min_v)/(max_v-min_v)
    img_inv = img_norm*(min_v-max_v) + max_v
    img_inv = img_inv.astype(origin_type)
    return img_inv


class DrawingEvaluator(object):

    def __init__(self, camera_indx=1, board_size=(0.56, 0.86), tag_size=0.09, scaling_factor=1000, drawing_topic='expected_drawing', visualize_expected_drawing=False):
        self.tag_names = ['tag_5', 'tag_6', 'tag_7']
        self.camera_indx = camera_indx
        self.board_x_size = board_size[0]
        self.board_y_size = board_size[1]
        self.tag_size = tag_size
        self.scaling_factor = scaling_factor # pixels per meter for unwarped image
        self.drawing_topic = drawing_topic
        self.visualize_expected_drawing = visualize_expected_drawing
        self.tf_listener = TF2Wrapper()
        self.marker_publisher = MarkerPublisher(self.drawing_topic)
        self.camera_parser = RealSenseCameraParser(camera_indx=camera_indx, verbose=False)
        self.camera_info_depth = self.camera_parser.get_camera_info_depth()
        self.camera_info_color = self.camera_parser.get_camera_info_color()
        self.camera_frame = 'camera_{}_link'.format(self.camera_indx)
        self.camera_optical_frame = 'camera_{}_color_optical_frame'.format(self.camera_indx)
        self.tag_frames = ['{}_{}'.format(tag_name, camera_indx) for tag_name in self.tag_names]
        self.projected_img_size = (self.scaling_factor*np.array([self.board_x_size, self.board_y_size])).astype(np.int32) # u,v (x,y)
        self.tag_pixel_size = int(1000 * self.tag_size)
        self.board_pixel_x_size = int(1000 * self.board_x_size)
        self.board_pixel_y_size = int(1000 * self.board_y_size)

    def _get_board_corners_bc(self):
        # return the board corner coordiantes in board coordinates
        # Board coordinates are centered at tag_names[0] and with the same xy orientation of the plane coordinates
        x_correction = 0.02
        board_corners_bc = np.array([
            [-0.5 * self.tag_size, 0.5 * self.tag_size, 0],
            [self.board_x_size - 0.5 * self.tag_size, 0.5 * self.tag_size, 0],
            [self.board_x_size - 0.5 * self.tag_size, 0.5 * self.tag_size - self.board_y_size + x_correction, 0],
            [-0.5 * self.tag_size, 0.5 * self.tag_size - self.board_y_size + x_correction, 0]
        ])
        return board_corners_bc

    def _filter_corners(self, img):
        patch_size = int(1.2 * self.tag_pixel_size)
        filtered_img = img.copy()
        max_value = np.max(img)
        filtered_img[:patch_size, :patch_size] = max_value
        filtered_img[:patch_size, self.board_pixel_x_size - patch_size:] = max_value
        filtered_img[self.board_pixel_y_size - patch_size:, :patch_size] = max_value
        return filtered_img

    def _get_board_coordinates_tf(self, ref_frame='med_base'):
        w_X_cf = self.tf_listener.get_transform(parent=ref_frame, child=self.camera_frame)
        cf_X_tags = [self.tf_listener.get_transform(parent=self.camera_frame, child=tag_frame) for tag_frame in self.tag_frames]
        w_X_tags = [w_X_cf @ cf_X_tag for cf_X_tag in cf_X_tags]

        # estimate plane normal in w
        tag_poses_w = [X[:3, 3] for X in w_X_tags]
        tag_plane_vectors_w = [pose_i - tag_poses_w[0] for pose_i in tag_poses_w[1:]]
        tag_plane_vectors_w = [pv / np.linalg.norm(pv) for pv in tag_plane_vectors_w]
        plane_normal_w = np.cross(tag_plane_vectors_w[1], tag_plane_vectors_w[0])

        # DO NOT TRUST THE ORIENTATION OF THE TAG SINCE IT CAN BE VERY NOISY
        # GET TF from world to board w_X_bc
        w_X_bc = np.eye(4)
        w_X_bc[:3, 0] = tag_plane_vectors_w[0]
        w_X_bc[:3, 2] = plane_normal_w
        w_X_bc[:3, 1] = np.cross(plane_normal_w, tag_plane_vectors_w[0])
        w_X_bc[:3, 3] = tag_poses_w[0]
        return w_X_bc

    def _tag_pixel_coordinates(self, img, pixel_uvs, axis_size=10, color=(255,0,0)):
        axis_dirs = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        axis = np.concatenate([np.zeros((1, 2), dtype=np.int32)] + [axis_dirs * (i + 1) for i in range(axis_size)], axis=0)
        pixel_uvs_ext = np.concatenate([pixel_uvs + axis_i for axis_i in axis], axis=0)
        img[pixel_uvs_ext[..., 1], pixel_uvs_ext[..., 0]] = np.array(color)
        return img

    def _get_pixel_coordinates(self, coords_cof, K=None, as_int=True):
        # Translate from camera optical frame coordinates to pixel coordinates
        if K is None:
            K = self.camera_info_color['K']
        coords_uvw = project_points_pinhole(coords_cof, K)
        coords_uv = coords_uvw[..., :2]
        if as_int:
            coords_uv = np.floor(coords_uv).astype(np.int32)
        return coords_uv

    def _compute_score(self, actual_drawing, desired_drawing):
        img_th = 50
        current_drawing_pixels = np.stack(np.where(actual_drawing > img_th), axis=-1)
        desired_drawing_pixels = np.stack(np.where(desired_drawing > img_th), axis=-1)
        tree = KDTree(current_drawing_pixels)
        min_dists, min_indxs = tree.query(desired_drawing_pixels)
        score = np.mean(min_dists)
        return score

    def publish_drawing_coordinates(self, drawing_coordinates, frame='med_base'):
        drawing_points = []
        for i, coord_i in enumerate(drawing_coordinates):
            point_i = Point()
            point_i.x = coord_i[0]
            point_i.y = coord_i[1]
            point_i.z = coord_i[2]
            drawing_points.append(point_i)
        self.marker_publisher.marker_type = self.marker_publisher.Marker.LINE_STRIP
        self.marker_publisher.marker_points = drawing_points
        self.marker_publisher.frame_id = frame
        self.marker_publisher.scale = [0.01, 0.01, 0.01]
        self.marker_publisher.data = np.zeros(7)


    def evaluate(self, expected_drawing_cooridnates, frame='med_base', save_path=None):
        """
        Evaluate the drawing
        Args:
            expecte_drawing_cooridnates:
            frame:
        Returns: score (0, inf).  The lower, the better.
        """
        # get image:
        color_img = self.camera_parser.get_image_color().copy()

        # estimate the board coorinates
        w_X_bc = self._get_board_coordinates_tf(ref_frame=frame)

        # Compute camera and tags tfs
        w_X_cf = self.tf_listener.get_transform(parent=frame, child=self.camera_frame)
        cf_X_cof = self.tf_listener.get_transform(parent=self.camera_frame, child=self.camera_optical_frame)
        w_X_cof = w_X_cf @ cf_X_cof
        cof_X_tags = [self.tf_listener.get_transform(parent=self.camera_optical_frame, child=tag_frame) for tag_frame in
                      self.tag_frames]

        board_corners_bc = self._get_board_corners_bc()

        board_corners_w = np.einsum('ij,kj->ki', w_X_bc[:3, :3], board_corners_bc) + w_X_bc[:3, 3]
        cof_X_bc = np.linalg.inv(w_X_cof) @ w_X_bc
        board_corners_cof = np.einsum('ij,kj->ki', cof_X_bc[:3, :3], board_corners_bc) + cof_X_bc[:3, 3]

        tag_poses_cof = np.stack([X[:3, 3] for X in cof_X_tags], axis=0)

        # Get the image coordinates of the board corners
        board_corners_uv = self._get_pixel_coordinates(board_corners_cof)
        tag_centers_uv = self._get_pixel_coordinates(tag_poses_cof)
        detected_color_img_q = color_img.copy()
        detected_color_img_q = self._tag_pixel_coordinates(detected_color_img_q, board_corners_uv, color=(255,0,0))
        detected_color_img_q = self._tag_pixel_coordinates(detected_color_img_q, tag_centers_uv, color=(0,255,0))

        # UNWARP The image
        destination_uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * np.repeat(
            np.expand_dims(self.projected_img_size, axis=0), 4, axis=0)
        H, _ = cv2.findHomography(board_corners_uv, destination_uv, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        unwarped_img = cv2.warpPerspective(color_img, H, tuple(self.projected_img_size), flags=cv2.INTER_LINEAR)

        # Binarize the image
        processed_unwarped_img = unwarped_img.copy()
        gray_img = cv2.cvtColor(processed_unwarped_img, cv2.COLOR_BGR2GRAY)
        th, gray_img_th_otsu = cv2.threshold(gray_img, 128, 192, cv2.THRESH_OTSU)
        # import pdb; pdb.set_trace()
        gray_img_th_otsu = self._filter_corners(gray_img_th_otsu)
        binarized_img = invert_img(gray_img_th_otsu)

        # Get the expected drawing coordinates in the rectified space:
        bc_X_w = np.linalg.inv(w_X_bc)
        drawing_bc = transform_points(expected_drawing_cooridnates, bc_X_w)
        drawing_bc[:,-1] = 0 # z = 0# force them to be on the board (on plane)
        drawing_cof = transform_points(drawing_bc, cof_X_bc)  # on camera optical frame coordiantes
        drawing_uvs = self._get_pixel_coordinates(drawing_cof, as_int=False)
        # Compute the expected image
        expected_img = np.zeros_like(binarized_img)
        drawing_uvs_ext = np.concatenate([drawing_uvs, np.ones((drawing_uvs.shape[0], 1))], axis=-1)
        drawing_uvs_rectified_uvw = np.einsum('ij,kj->ki', H, drawing_uvs_ext)
        drawing_uvs_rectified = (drawing_uvs_rectified_uvw/drawing_uvs_rectified_uvw[...,-1:].repeat(3, axis=-1))[..., :2] # Normalize
        drawing_uvs_rectified = np.clip(
            np.rint(drawing_uvs_rectified),
            np.zeros(2), np.flip(expected_img.shape[:2]) - 1).astype(np.int32)

        expected_img[drawing_uvs_rectified[..., 1], drawing_uvs_rectified[..., 0]] = 255  # paint it white

        # view desired drawing on the image
        expected_drawing_img = color_img.copy()
        drawing_uvs_int = np.clip(np.rint(drawing_uvs), np.zeros(2),
                                  np.flip(expected_drawing_img.shape[:2]) - 1).astype(np.int32)
        expected_drawing_img[drawing_uvs_int[..., 1], drawing_uvs_int[..., 0]] = np.array(
            [255, 165, 0])  # paint it orange

        # compute score
        score = self._compute_score(binarized_img, expected_img)

        # save in case save_path is not None
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                print('created: ', save_path)
            file_name='drawing_evaluation_{}.png'
            full_path = os.path.join(save_path, file_name)
            # original
            plt.figure(1)
            plt.imshow(color_img)
            plt.savefig(full_path.format('original'))
            # detected
            plt.figure(2)
            plt.imshow(detected_color_img_q)
            plt.savefig(full_path.format('detected'))
            # unwarped
            plt.figure(3)
            plt.imshow(unwarped_img)
            plt.savefig(full_path.format('unwarped'))
            # binarized
            plt.figure(4)
            plt.imshow(binarized_img)
            plt.savefig(full_path.format('binarized'), dpi=400)
            # expected drawing
            plt.figure(5)
            plt.imshow(expected_drawing_img)
            plt.savefig(full_path.format('expected_overlapped'), dpi=400)
            # expected binarized projected
            plt.figure(6)
            plt.imshow(expected_img)
            plt.savefig(full_path.format('expected_binarized_unwarped'), dpi=400)

        return score, binarized_img, expected_img


# DEBUG: --

if __name__ == '__main__':
    rospy.init_node('drawing_eval_testing')
    evaluator = DrawingEvaluator()

    # expected_drawing_coorinates:
    num_points = 1000
    start_point = [0.55, 0.2]
    edc_x = start_point[0] * np.ones((num_points,))
    edc_y = np.linspace(start_point[1] - 0.55, start_point[1], num=num_points)
    edc_z = 0.01 * np.ones((num_points,))
    expected_drawing_cooridnates = np.stack([edc_x, edc_y, edc_z], axis=-1)

    # evaluate
    score, actual_drawing, expected_drawing = evaluator.evaluate(expected_drawing_cooridnates,
                                                                      frame='med_base',
                                                                      save_path='/home/mmint/Desktop/eval_test')
    print('SCORE: ', score)


