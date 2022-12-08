
import torch.nn as nn
import torch.nn.functional as F
from manipulation_via_membranes.bubble_learning.models.pointnet.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction

class PointNet2ClsBase(nn.Module):

    def __init__(self, normal_channel=True):
        """
        NOTE: The input pointcloud must have more than 128 points.
        Args:
            normal_channel:
        """
        super().__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        #  npoint, radius_list, nsample_list, in_channel, mlp_list
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)

    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1)
        B, num_channels, num_points = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        return x, l3_points

    @classmethod
    def get_name(cls):
        return 'pointnet2_cls_msg_base'

    @property
    def name(self):
        return self.get_name()


class PointNet2ClsMsg(PointNet2ClsBase):
    def __init__(self, num_class, normal_channel=True):
        super().__init__(normal_channel=normal_channel)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        x, l3_points = super().forward(xyz)
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x, l3_points

    @classmethod
    def get_name(cls):
        return 'pointnet2_cls_msg'


class PointNet2ObjectEmbedding(PointNet2ClsBase):
    def __init__(self, obj_embedding_size, normal_channel=True):
        self.obj_embedding_size = obj_embedding_size
        super().__init__(normal_channel=normal_channel)
        self.embedding_fc = nn.Linear(256, self.obj_embedding_size)

    def forward(self, xyz):
        x, l3_points = super().forward(xyz)
        x = self.embedding_fc(x)
        return x

    @classmethod
    def get_name(cls):
        return 'pointnet2_obj_embedding'
