import numpy as np
import torch
import cv2
import torchvision

from manipulation_via_membranes.bubble_learning.aux.orientation_trs import QuaternionToAxis


def get_pose_images(trans_pred, rot_angle_pred, trans_gth, rot_angle_gth):
    images = []
    for i in range(len(trans_pred)):
        img = np.zeros([100, 100, 3], dtype=np.uint8)
        img.fill(100)
        pred_param = find_rect_param(trans_pred[i], rot_angle_pred[i], img)
        color_p = (255, 0, 0)
        draw_angled_rec(*pred_param, color_p, img)
        gth_param = find_rect_param(trans_gth[i], rot_angle_gth[i], img)
        color_gth = (0, 0, 255)
        draw_angled_rec(*gth_param, color_gth, img)
        img = torch.tensor(img)
        img = img.permute(2, 0, 1)
        images.append(img)
    return images


def get_pose_images_grid(trans_pred, rot_angle_pred, trans_gth, rot_angle_gth):
    pose_images = get_pose_images(trans_pred, rot_angle_pred, trans_gth, rot_angle_gth)
    pose_grid = torchvision.utils.make_grid(pose_images)
    return pose_grid


def get_object_pose_images_grid(obj_pose_pred, obj_pose_gth, plane_normal):
    obj_trans_pred = obj_pose_pred[..., :3]
    obj_rot_pred = obj_pose_pred[..., 3:]
    obj_rot_angle_pred = get_angle_from_axis_angle(obj_rot_pred, plane_normal)
    obj_trans_gth = obj_pose_gth[..., :3]
    obj_rot_gth = obj_pose_gth[..., 3:]
    obj_rot_angle_gth = get_angle_from_axis_angle(obj_rot_gth, plane_normal)
    pose_grid = get_pose_images_grid(obj_trans_pred, obj_rot_angle_pred, obj_trans_gth, obj_rot_angle_gth)
    # they should match the bottom imprint (left one)
    return pose_grid


# ------ AUX FUNCTIONS ----


def get_angle_from_axis_angle(orientation, plane_normal):
    if orientation.shape[-1] == 4:
        q_to_ax = QuaternionToAxis()
        axis_angle = torch.from_numpy(q_to_ax._tr(orientation.detach().numpy()))
    else:
        axis_angle = orientation
    projection = torch.einsum('bi,i->b', axis_angle, plane_normal)
    normal_axis_angle = projection.unsqueeze(-1) * plane_normal.unsqueeze(0)
    angle = torch.norm(normal_axis_angle, dim=-1) * torch.sign(projection) + np.pi*0.5
    return angle


def find_rect_param(trans, rot, img):
    height = 0.06 * 100 / 0.15
    width = 0.015 * 100 / 0.15
    center_x = img.shape[0] / 2 + trans[0] * 10 / 0.15
    center_y = img.shape[1] / 2 + trans[1] * 10 / 0.15
    return center_x, center_y, width, height, rot.item()


def draw_angled_rec(x0, y0, width, height, angle, color, img):
    b = np.cos(angle) * 0.5
    a = np.sin(angle) * 0.5
    pt0 = (int(x0 - a * height - b * width),
           int(y0 + b * height - a * width))
    pt1 = (int(x0 + a * height - b * width),
           int(y0 - b * height - a * width))
    pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
    pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

    cv2.line(img, pt0, pt1, color, 3)
    cv2.line(img, pt1, pt2, color, 3)
    cv2.line(img, pt2, pt3, color, 3)
    cv2.line(img, pt3, pt0, color, 3)


