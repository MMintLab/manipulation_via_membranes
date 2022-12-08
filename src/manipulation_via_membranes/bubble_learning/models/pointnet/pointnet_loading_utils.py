import os
import sys
import torch

import manipulation_via_membranes.bubble_learning.models.pointnet as pointnet_pkg
from manipulation_via_membranes.bubble_learning.models.pointnet.pointnet2_cls_msg import PointNet2ClsMsg, PointNet2ObjectEmbedding
from manipulation_via_membranes.bubble_learning.models.pointnet.pointnet_classifier import PointNetClassifier


pointnet_pkg_path = os.path.dirname(os.path.abspath(pointnet_pkg.__file__))


def get_checkpoints_path():
    checkpoints_path = os.path.join(pointnet_pkg_path, 'checkpoints')
    return checkpoints_path


def load_pointnet_model(pointnet_model, freeze=False, partial_load=False, pretrained_model_name=None):
    if pretrained_model_name is None:
        pretrained_model_name = pointnet_model.name
    checkpoint_name = '{}_best_model.pth'.format(pretrained_model_name)
    checkpoint_path = os.path.join(get_checkpoints_path(), checkpoint_name)
    checkpoint = torch.load(checkpoint_path)
    # load the checkpoint
    if partial_load:
        # load only the weigths that match the model
        model_dict = pointnet_model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict} # filter out params not in model_dict
        model_dict.update(pretrained_dict) # update model_dict with the values we loaded from checkpoint
        pointnet_model.load_state_dict(model_dict) # load
    else:
        # load all state_dict
        pointnet_model.load_state_dict(checkpoint)
    if freeze:
        for param_name, param in pointnet_model.named_parameters():
            if partial_load:
                if param_name in checkpoint['model_state_dict']:
                    param.requires_grad = False
            else:
                param.requires_grad = False
    return pointnet_model


def get_pretrained_pointnet2_cls_msg_best_model(freeze=False, partial_load=False):
    model = PointNet2ClsMsg(num_class=40, normal_channel=False)
    model = load_pointnet_model(model, freeze=freeze, partial_load=partial_load)
    return model


def get_pretrained_pointnet2_object_embeding(obj_embedding_size=10, freeze=False):
    model = PointNet2ObjectEmbedding(obj_embedding_size=obj_embedding_size, normal_channel=False)
    model = load_pointnet_model(model, freeze=freeze, partial_load=True, pretrained_model_name=PointNet2ClsMsg.get_name())
    return model


def get_pretrained_pointnet_classifier(freeze=False, partial_load=True):
    model = PointNetClassifier()
    model = load_pointnet_model(model, freeze=freeze, partial_load=partial_load)
    if freeze:
        model.eval()
    return model

# DEBUG:
if __name__ == '__main__':
    object_embedding_model = get_pretrained_pointnet2_object_embeding(obj_embedding_size=10)
    points = torch.ones((10, 129, 3)) # NOTE. the
    out, _ = object_embedding_model(points)
