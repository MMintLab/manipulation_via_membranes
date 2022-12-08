import numpy as np
from mmint_camera_utils.camera_utils.point_cloud_utils import pack_o3d_pcd, view_pointcloud
import copy
from numpy.random import default_rng
from manipulation_via_membranes.bubble_pivoting.aux.load_confs import save_object_models, load_object_params
from scipy.spatial.transform import Rotation as R
import sys
import argparse

def generate_rolling_pin(width, length, tip_height, num_points=160):
    radius = width / 2
    point_per_circle = int(num_points*0.06)
    cylinder_points = np.floor(num_points * length/(length+2*tip_height))
    num_circles = int(np.floor(cylinder_points/point_per_circle))
    x_values = np.linspace(-0.5*length, 0.5*length, num_circles)
    angles = (2*np.pi)/point_per_circle * np.arange(point_per_circle)
    circle_yz_unit = np.stack([np.cos(angles), np.sin(angles)],axis=-1)    
    circles_yz_unit = np.repeat(np.expand_dims(circle_yz_unit,0), num_circles ,axis=0)
    x_values = np.repeat(np.expand_dims(x_values,[-2,-1]), point_per_circle,axis=-2)
    circles_yz = radius * circles_yz_unit
    circles_xyz = np.concatenate([x_values, circles_yz], axis=-1).reshape(-1,3)
    tip_points = np.ceil(num_points * tip_height/(length+tip_height))
    num_circles_tip = int(np.floor(tip_points/point_per_circle))
    tip_x_values = np.linspace(0.5*length, 0.5*length+tip_height, num_circles_tip)
    tip_x_values = np.repeat(np.expand_dims(tip_x_values,[-2,-1]), point_per_circle, axis=-2)
    tip_heights  = np.linspace(0, tip_height, num_circles_tip)
    tip_radiis = np.sqrt(radius**2 - tip_heights**2)
    tip_radiis = np.expand_dims(tip_radiis, [-2,-1])
    circles_tip_yz_unit = np.repeat(np.expand_dims(circle_yz_unit,0), num_circles_tip ,axis=0)
    circles_tip_yz = tip_radiis * circles_tip_yz_unit
    circles_tip_xyz = np.concatenate([tip_x_values, circles_tip_yz], axis=-1).reshape(-1,3)
    circles_tip_xyz_neg = copy.deepcopy(circles_tip_xyz)
    circles_tip_xyz_neg[:, 0] *= -1
    object_points = np.concatenate([circles_xyz, circles_tip_xyz, circles_tip_xyz_neg], axis=0)
    # Filter points to have num_points
    object_points_filtered = filter_object_points(object_points, num_points)

    rolling_pin_model = np.concatenate([object_points_filtered, np.zeros_like(object_points_filtered)], axis=-1)
    # view_pointcloud(rolling_pin_model)
    return rolling_pin_model
     
def generate_stick(width, length, num_points=150):
    radius = width / 2
    point_per_circle = int(num_points*0.06)
    num_circles = int(np.floor(num_points/point_per_circle))
    x_values = np.linspace(-0.5*length, 0.5*length, num_circles)
    angles = (2*np.pi)/point_per_circle * np.arange(point_per_circle)
    circle_yz_unit = np.stack([np.cos(angles), np.sin(angles)],axis=-1)    
    circles_yz_unit = np.repeat(np.expand_dims(circle_yz_unit,0), num_circles ,axis=0)
    x_values = np.repeat(np.expand_dims(x_values,[-2,-1]), point_per_circle,axis=-2)
    circles_yz = radius * circles_yz_unit
    circles_xyz = np.concatenate([x_values, circles_yz], axis=-1).reshape(-1,3)
    object_points = circles_xyz
    # Filter points to have num_points
    object_points_filtered = filter_object_points(object_points, num_points)

    stick_model = np.concatenate([object_points_filtered, np.zeros_like(object_points_filtered)], axis=-1)
    # view_pointcloud(stick_model)
    return stick_model

def generate_double_marker(width_1, length_1, width_2, length_2, num_points=150):
    radius_1 = width_1 / 2
    point_per_circle = int(num_points*0.06)
    num_points_1 = num_points * length_1 / (length_1 + length_2)
    num_circles_1 = int(np.floor(num_points_1/point_per_circle))
    x_values_1 = np.linspace(-0.5*(length_1+length_2), 0.5*(length_1-length_2), num_circles_1)
    x_values_1 = np.repeat(np.expand_dims(x_values_1,[-2,-1]), point_per_circle,axis=-2)
    angles = (2*np.pi)/point_per_circle * np.arange(point_per_circle)
    circle_yz_unit = np.stack([np.cos(angles), np.sin(angles)],axis=-1)    
    circles_yz_unit_1 = np.repeat(np.expand_dims(circle_yz_unit,0), num_circles_1 ,axis=0)
    circles_yz_1 = radius_1 * circles_yz_unit_1
    circles_xyz_1 = np.concatenate([x_values_1, circles_yz_1], axis=-1).reshape(-1,3)

    radius_2 = width_2 / 2
    num_points_2 = num_points * length_2 / (length_1 + length_2)
    num_circles_2 = int(np.floor(num_points_2/point_per_circle))
    x_values_2 = np.linspace(0.5*(length_1-length_2), 0.5*(length_1+length_2), num_circles_2)
    x_values_2 = np.repeat(np.expand_dims(x_values_2,[-2,-1]), point_per_circle,axis=-2)
    angles = (2*np.pi)/point_per_circle * np.arange(point_per_circle)
    circles_yz_unit_2 = np.repeat(np.expand_dims(circle_yz_unit,0), num_circles_2 ,axis=0)
    circles_yz_2 = radius_2 * circles_yz_unit_2
    circles_xyz_2 = np.concatenate([x_values_2, circles_yz_2], axis=-1).reshape(-1,3)

    object_points = np.concatenate([circles_xyz_1, circles_xyz_2], axis=0)
    # Filter points to have num_points
    object_points_filtered = filter_object_points(object_points, num_points)

    double_marker_model = np.concatenate([object_points_filtered, np.zeros_like(object_points_filtered)], axis=-1)
    # view_pointcloud(double_marker_model)
    return double_marker_model    

def generate_double_stick(width, length, num_points=150):
    radius = width / 2
    num_points_single = int(num_points/2)
    point_per_circle = int(num_points_single*0.06)
    num_circles = int(np.floor(num_points_single/point_per_circle))
    x_values = np.linspace(-0.5*length, 0.5*length, num_circles)
    angles = (2*np.pi)/point_per_circle * np.arange(point_per_circle)
    circle_yz_unit = np.stack([np.cos(angles), np.sin(angles)],axis=-1)    
    circles_yz_unit = np.repeat(np.expand_dims(circle_yz_unit,0), num_circles ,axis=0)
    x_values = np.repeat(np.expand_dims(x_values,[-2,-1]), point_per_circle,axis=-2)
    circles_yz = radius * circles_yz_unit
    circles_xyz = np.concatenate([x_values, circles_yz], axis=-1).reshape(-1,3)
    right_stick = copy.deepcopy(circles_xyz)
    right_stick[:, 1] += radius
    left_stick = copy.deepcopy(circles_xyz)
    left_stick[:,1] -= radius
    object_points = np.concatenate([right_stick, left_stick], axis=0)
    # Filter points to have num_points
    object_points_filtered = filter_object_points(object_points, num_points)

    stick_model = np.concatenate([object_points_filtered, np.zeros_like(object_points_filtered)], axis=-1)
    # view_pointcloud(stick_model)
    return stick_model

def generate_spoon(handle_width, handle_length, triangle_width, triangle_length, circle_radius, intersection_length, num_points=170):
    radius = handle_width / 2
    total_length = handle_length + triangle_length - intersection_length + circle_radius
    num_points_handle = num_points * handle_length * 0.4 / total_length
    point_per_circle = int(np.ceil(num_points_handle*0.01))
    num_circles = int(np.ceil(num_points_handle/point_per_circle))
    handle_end = 0.5*total_length-circle_radius-triangle_length+intersection_length
    x_values = np.linspace(-0.5*total_length, handle_end, num_circles)
    angles = (2*np.pi)/point_per_circle * np.arange(point_per_circle)
    circle_yz_unit = np.stack([np.cos(angles), np.sin(angles)],axis=-1)    
    circles_yz_unit = np.repeat(np.expand_dims(circle_yz_unit,0), num_circles ,axis=0)
    x_values = np.repeat(np.expand_dims(x_values,[-2,-1]), point_per_circle,axis=-2)
    circles_yz = radius * circles_yz_unit
    circles_xyz = np.concatenate([x_values, circles_yz], axis=-1).reshape(-1,3)
    handle_points = circles_xyz

    # Triangle
    num_points_triangle = num_points * (triangle_length-intersection_length) / total_length
    num_rows = int(np.ceil((-1 + np.sqrt(1 + 8 * num_points_triangle/2)) / 2))
    triangle_start = handle_end - intersection_length
    triangle_points = None
    for i in np.arange(num_rows):
        x = i * triangle_length / (num_rows-1)
        if x > intersection_length:
            y_max = x * (triangle_width/2) / triangle_length
            new_y_values = np.linspace(-y_max, y_max, i+1)
            new_x_values = np.repeat(x + triangle_start, i+1) 
            new_z_values = np.repeat(handle_width/2, i+1)
            new_points_above = np.stack([new_x_values, new_y_values, new_z_values]).transpose()
            new_points_bellow = np.stack([new_x_values, new_y_values, -new_z_values]).transpose()
            points_border = int(point_per_circle/4)
            z_values_border = np.linspace(-handle_width/2, handle_width/2, points_border)
            new_points_border_left = np.stack([np.repeat(x + triangle_start, points_border),  np.repeat(y_max, points_border), z_values_border]).transpose()
            new_points_border_right = np.stack([np.repeat(x + triangle_start, points_border),  np.repeat(-y_max, points_border), z_values_border]).transpose()
            if triangle_points is None:
                triangle_points = np.concatenate([new_points_above, new_points_bellow, new_points_border_left, new_points_border_right], axis=0)
            else:
                triangle_points = np.concatenate([triangle_points, new_points_above, new_points_bellow, new_points_border_left, new_points_border_right], axis=0)

    # Semicircle
    num_points_circle = num_points * circle_radius * 4 / total_length
    num_rows = int(np.ceil((-1 + np.sqrt(1 + 8 * num_points_circle/2)) / 4))
    circle_start = handle_end - intersection_length + triangle_length
    circle_points = None
    for i in np.arange(num_rows-1):
        x = (i+1) * circle_radius / (num_rows-1)
        y_max = np.sqrt(circle_radius**2 - x**2)
        points_row = int(np.ceil(np.sqrt(2*num_points_circle/np.pi-x**2)))
        new_y_values = np.linspace(-y_max, y_max, points_row)
        new_x_values = np.repeat(x + circle_start, points_row) 
        new_z_values = np.repeat(handle_width/2, points_row)
        new_points_above = np.stack([new_x_values, new_y_values, new_z_values]).transpose()
        new_points_bellow = np.stack([new_x_values, new_y_values, -new_z_values]).transpose()
        points_border = int(np.ceil(point_per_circle/4))
        z_values_border = np.linspace(-handle_width/2, handle_width/2, points_border)
        new_points_border_left = np.stack([np.repeat(x + circle_start, points_border),  np.repeat(y_max, points_border), z_values_border]).transpose()
        new_points_border_right = np.stack([np.repeat(x + circle_start, points_border),  np.repeat(-y_max, points_border), z_values_border]).transpose()
        if circle_points is None:
            circle_points = np.concatenate([new_points_above, new_points_bellow, new_points_border_left, new_points_border_right], axis=0)
        else:
            circle_points = np.concatenate([circle_points, new_points_above, new_points_bellow, new_points_border_left, new_points_border_right], axis=0)
    object_points = np.concatenate([handle_points, triangle_points, circle_points], axis=0)
    object_points_filtered = filter_object_points(object_points, num_points=num_points)
    spoon_model = np.concatenate([object_points_filtered, np.zeros_like(object_points_filtered)], axis=-1)
    # view_pointcloud(spoon_model, frame=True)
    return spoon_model

# def generate_truncated_triangle(triangle_start, triangle_length, triangle_width_big, triangle_width_small, depth, points_border, inverse, num_points):
#     # Triangle
#     import pdb; pdb.set_trace()
#     num_rows_total = int((-1 + np.sqrt(1 + 8 * num_points/2)) / 2)
#     truncated_height = triangle_length * triangle_width_small / (triangle_width_big - triangle_width_small)
#     triangle_points = None
#     import pdb; pdb.set_trace()
#     for i in np.arange(num_rows):
#         x = i * (triangle_length+truncated_height) / (num_rows-1)
#         if x > truncated_height:
#             y_max = x * (triangle_width_big/2) / triangle_length
#             new_y_values = np.linspace(-y_max, y_max, i+1)
#             new_x_values = np.repeat(x + triangle_start, i+1) 
#             new_z_values = np.repeat(depth/2, i+1)
#             new_points_above = np.stack([new_x_values, new_y_values, new_z_values]).transpose()
#             new_points_bellow = np.stack([new_x_values, new_y_values, -new_z_values]).transpose()
#             z_values_border = np.linspace(-depth/2, depth/2, points_border)
#             new_points_border_left = np.stack([np.repeat(x + triangle_start, points_border),  np.repeat(y_max, points_border), z_values_border]).transpose()
#             new_points_border_right = np.stack([np.repeat(x + triangle_start, points_border),  np.repeat(-y_max, points_border), z_values_border]).transpose()
#             if triangle_points is None:
#                 triangle_points = np.concatenate([new_points_above, new_points_bellow, new_points_border_left, new_points_border_right], axis=0)
#             else:
#                 triangle_points = np.concatenate([triangle_points, new_points_above, new_points_bellow, new_points_border_left, new_points_border_right], axis=0)
#     return triangle_points

def create_rectangle_pc(length, height, center, rotation, density, color=np.array([0,0,0])):
    one_dim_density = np.sqrt(density)
    grid_hor, grid_vert = np.meshgrid(np.linspace(-length/2, length/2, int(np.ceil(one_dim_density*length))),
                                    np.linspace(-height/2, height/2, int(np.ceil(one_dim_density*height))))
    points_hor = grid_hor.flatten()
    points_vert = grid_vert.flatten()
    rectangle = np.stack([np.zeros_like(points_hor), points_hor, points_vert], axis=1)
    r = R.from_quat(rotation)
    rectangle = rectangle @ r.as_matrix()
    rectangle += center
    rectangle_pc = np.concatenate([rectangle, np.zeros_like(rectangle)], axis=1)
    rectangle_pc[:,3:] += color
    return rectangle_pc

def create_rectangular_prism(height, width, depth, num_points, color=np.array([0,0,0])):
    area = 2*(height*width + width*depth + depth*height)
    density = num_points/area
    front = create_rectangle_pc(length=width, height=height, center=np.array([depth/2, 0, 0]), rotation=np.array([1,0,0,0]), density=density, color=color)
    back = create_rectangle_pc(length=width, height=height, center=np.array([-depth/2, 0, 0]), rotation=np.array([1,0,0,0]), density=density, color=color)
    right = create_rectangle_pc(length=depth, height=height, center=np.array([0, width/2, 0]), rotation=np.array([np.sqrt(2)/2,np.sqrt(2)/2,0,0]), density=density, color=color)
    left = create_rectangle_pc(length=depth, height=height, center=np.array([0, -width/2, 0]), rotation=np.array([np.sqrt(2)/2,np.sqrt(2)/2,0,0]), density=density, color=color)
    up = create_rectangle_pc(length=width, height=depth, center=np.array([0, 0, height/2]), rotation=np.array([np.sqrt(2)/2,0,np.sqrt(2)/2,0]), density=density, color=color)
    down = create_rectangle_pc(length=width, height=depth, center=np.array([0, 0, -height/2]), rotation=np.array([np.sqrt(2)/2,0,np.sqrt(2)/2,0]), density=density, color=color)
    faces = np.concatenate([front, back, right, left, up, down], axis=0)
    return faces


def generate_spatula(handle_width, handle_length, handle_depth, tip_width, tip_length, tip_depth, num_points=100):
    handle_area = handle_length * handle_width
    tip_area = tip_length * tip_width
    total_area = handle_area + tip_area
    num_points_handle = int(np.ceil(num_points*handle_area/total_area))
    handle_model = create_rectangular_prism(handle_depth, handle_width, handle_length, num_points_handle)
    tip_model = create_rectangular_prism(tip_depth, tip_width, tip_length, num_points-num_points_handle)
    tip_model[:,0] += (handle_length + tip_length)/2
    object_model = np.concatenate([handle_model, tip_model], axis=0)
    object_model_filtered = filter_object_points(object_model, num_points=num_points)
    # view_pointcloud(object_model_filtered, frame=True)
    return object_model_filtered


def filter_object_points(object_points, num_points=3000):
    num_points = 144
    raw_num_points = object_points.shape[0]
    print(raw_num_points)
    if raw_num_points > num_points:
        indices = default_rng().choice(raw_num_points, size=num_points, replace=False)
        object_points_filtered = object_points[indices]
    else:
        indices = default_rng().choice(raw_num_points, size=num_points-raw_num_points, replace=False)
        object_points_filtered = np.concatenate([object_points, object_points[indices]], axis=0)
    return object_points_filtered


def create_pivoting_object_models(num_points=3000, view=False):
    pivoting_object_models = {}
    obj_params_dicc = load_object_params()['objects']
    for k, v in obj_params_dicc.items():
        function = getattr(sys.modules[__name__], v['function'])
        params = v['params']
        for p, s in params.items():
            params[p] = float(s)
        object_model_i = function(**params, num_points=num_points)
        if view:
            print(k)
            view_pointcloud(object_model_i)
        pivoting_object_models[k] = pack_o3d_pcd(object_model_i)
    return pivoting_object_models

# Save them:

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pivoting object models')
    parser.add_argument('num_points', type=int, help='Number of points for each model')
    parser.add_argument('--view', type=bool, default=False, help='View pointclouds while creating them')
    args = parser.parse_args()
    num_points = args.num_points
    view = args.view
    models = create_pivoting_object_models(num_points, view)
    save_object_models(models)
    
