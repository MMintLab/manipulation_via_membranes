import yaml
import os
import numpy as np
import pandas as pd

from mmint_camera_utils.camera_utils.point_cloud_utils import pack_o3d_pcd, unpack_o3d_pcd

package_path = project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('/bubble_control')[0], 'bubble_control')


def _load_config_from_path(path):
    config = None
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


def load_bubble_reconstruction_params():
    bubble_reconstruction_configs_path = os.path.join(package_path, 'config', 'bubble_reconstruction_params.yaml')
    bubble_reconstruction_configs = _load_config_from_path(bubble_reconstruction_configs_path)
    return bubble_reconstruction_configs


def load_plane_params():
    bubble_reconstruction_configs_path = os.path.join(package_path, 'config', 'plane_params.yaml')
    bubble_reconstruction_configs = _load_config_from_path(bubble_reconstruction_configs_path)
    return bubble_reconstruction_configs


def load_object_models():
    bubble_icp_models_path = os.path.join(package_path, 'config', 'object_models.npy')
    with open(bubble_icp_models_path, 'rb') as f:
        object_models = np.load(f, allow_pickle=True).item()

    # pack object models as pcd
    for k, ar_i in object_models.items():
        object_models[k] = pack_o3d_pcd(ar_i)
    return object_models


def save_object_models(object_models_dict):
    bubble_icp_models_path = os.path.join(package_path, 'config', 'object_models.npy')
    # unpack object models to numpy arrays
    for k,pcd_i in object_models_dict.items():
        object_models_dict[k] = unpack_o3d_pcd(pcd_i)
    with open(bubble_icp_models_path, 'wb') as f:
        np.save(f, object_models_dict)


def load_marker_params():
    marker_params_path = os.path.join(package_path, 'config', 'marker_params.csv')
    marker_params_df = pd.read_csv(marker_params_path)
    return marker_params_df
