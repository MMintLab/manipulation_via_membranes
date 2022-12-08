import numpy as np
import torch
import torch.nn as nn


class ModelPoseLoss(torch.nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        if criterion is None:
            self.criterion = torch.nn.MSELoss()

    def forward(self, R_1, t_1, R_2, t_2, model_points):
        m_1 = self._transform_model_points(R_1, t_1, model_points)
        m_2 = self._transform_model_points(R_2, t_2, model_points)
        loss = 0.5 * self.criterion(m_1, m_2)
        return loss

    def _transform_model_points(self, R, t, model_points):
        num_points = model_points.shape[-2] # (batch, num_points, space_size)
        m_rot = torch.einsum('...jk,...mk->...mj', R, model_points)
        m_tr = m_rot + t.unsqueeze(-2).repeat_interleave(num_points, dim=-2)
        return m_tr


class PoseLoss(ModelPoseLoss):
    def __init__(self, object_points, criterion=None):
        super().__init__(criterion=criterion)
        model_t = torch.tensor(object_points, dtype=torch.float)
        self.model = nn.Parameter(model_t, requires_grad=False)
        self.num_points = self.model.shape[0]

    def forward(self, R_1, t_1, R_2, t_2):
        # import pdb; pdb.set_trace()
        m_1 = self._transform_model(R_1, t_1)
        m_2 = self._transform_model(R_2, t_2)
        loss = 0.5 * self.criterion(m_1, m_2)
        return loss

    def _transform_model(self, R, t):
        # keep it for compatibility
        m_rot = torch.einsum('...jk,...mk->...mj', R, self.model)
        m_tr = m_rot + t.unsqueeze(-2).repeat_interleave(self.num_points, dim=-2)
        return m_tr


class BoxPoseLoss(ModelPoseLoss):
    pass


class PlanarBoxPoseLoss(torch.nn.Module):

    def __init__(self, box_size, device=None, criterion=None):
        super().__init__()
        self.box_size = box_size # (2,) np.array containing half sizes of the rectangular object
        self.device = device
        self.criterion = criterion
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if criterion is None:
            self.criterion = torch.nn.MSELoss()
        self.model = self._get_model(self.box_size)
        self.num_points = self.model.shape[0]

    def _get_model(self, box_size):
        base_points = [
            [0, 0],
            [0, 1],
            [1, 1],
            [0, 1],
            [-1, 1],
            [-1, 0],
            [-1, -1],
            [0, -1],
            [1, -1]
        ]
        base_points = box_size*np.array(base_points)
        base_points_t = torch.tensor(base_points, dtype=torch.float).to(self.device)
        return base_points_t

    def forward(self, pose1, pose2):
        """
        Assumptions:
            - pose composed by [x,y, theta]
        """
        xy_1 = pose1[..., :2]
        theta_1 = pose1[..., 2]
        xy_2 = pose2[..., :2]
        theta_2 = pose2[..., 2]
        m1 = self._transform_model(xy_1, theta_1)
        m2 = self._transform_model(xy_2, theta_2)
        # loss = 2*self.criterion(m1, m2)
        loss = self.criterion(torch.norm(m1-m2, dim=-1), torch.norm(m1-m1, dim=-1))
        # import pdb; pdb.set_trace()
        return loss

    def _transform_model(self, xy, theta):
        ctheta = torch.cos(theta)
        stheta = torch.sin(theta)
        R1x = torch.stack([ctheta, -stheta], dim=-1)
        R2x = torch.stack([stheta, ctheta], dim=-1)
        R = torch.stack([R1x, R2x], dim=-2)
        # import pdb; pdb.set_trace()
        m_rot = torch.einsum('...jk,mk->...mj', R, self.model)
        t = torch.stack(self.num_points*[xy],dim=-2)
        m_tr = m_rot + t
        return m_tr


if __name__ == '__main__':
    # Debug
    import gnureadline
    box_size = np.array([0.025, 0.05])
    pbp_loss = PlanarBoxPoseLoss(box_size=box_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    N = 10
    # theta_1 = np.random.uniform(0,2*np.pi, N)
    theta_1 = np.zeros((N,5))
    xy_1 = np.zeros((N,5,2))
    # xy_1 = np.random.uniform(-2,2,(N,2))
    pose1_ar = np.insert(xy_1, -1, theta_1, axis=-1)
    pose1 = torch.tensor(pose1_ar, dtype=torch.float).to(device)
    pose2 = torch.tensor(pose1_ar+np.array([1,1,0]), dtype=torch.float).to(device) # copy the same so the loss is 0
    loss_i = pbp_loss(pose1, pose2)
    print('loss:', loss_i)
