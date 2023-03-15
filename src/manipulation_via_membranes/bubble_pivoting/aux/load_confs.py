import yaml
import os
import numpy as np
import json
from mmint_tools.camera_tools.pointcloud_utils import pack_o3d_pcd, unpack_o3d_pcd

package_path = project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('/bubble_pivoting')[0], 'bubble_pivoting')


def _load_config_from_path(path):
    config = None
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


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
    for k, pcd_i in object_models_dict.items():
        object_models_dict[k] = unpack_o3d_pcd(pcd_i)
    with open(bubble_icp_models_path, 'wb') as f:
        np.save(f, object_models_dict)


def load_object_params():
    object_params_path = os.path.join(package_path, 'config', 'object_params.json')
    with open(object_params_path) as json_file:
        object_params_dicc = json.load(json_file)
    return object_params_dicc