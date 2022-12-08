import os


def load_model_version(Model, data_name, load_version):
    model_name = Model.get_name()
    version_chkp_path = os.path.join(data_name, 'tb_logs', '{}'.format(model_name),
                                     'version_{}'.format(load_version), 'checkpoints')
    checkpoints_fs = [f for f in os.listdir(version_chkp_path) if
                      os.path.isfile(os.path.join(version_chkp_path, f))]
    checkpoint_path = os.path.join(version_chkp_path, checkpoints_fs[0])

    model = Model.load_from_checkpoint(checkpoint_path, dataset_params={'data_name': data_name})
    return model