import torch
import numpy as np
import tf.transformations as tr
import pytorch3d.transforms as batched_trs


class QuaternionToAxis(object):

    def __init__(self, keys_to_tr=None):
        self.keys_to_tr = keys_to_tr

    def __call__(self, sample):
        if self.keys_to_tr is None:
            # transform all that has quat in the key
            for k, v in sample.items():
                if 'quat' in k:
                    sample[k] = self._tr(v)
        else:
            for key in self.keys_to_tr:
                if key in sample:
                    sample[key] = self._tr(sample[key])
        return sample

    def inverse(self, sample):
        # apply the inverse transformation
        if self.keys_to_tr is None:
            # trasform all that has quat in the key
            for k, v in sample.items():
                if 'quat' in k:
                    sample[k] = self._tr_inv(v)
        else:
            for key in self.keys_to_tr:
                if key in sample:
                    sample[key] = self._tr_inv(sample[key])
        return sample

    @classmethod
    def _tr(cls, x):
        # transform a quaternion encoded rotation to an axis one with 3 values representing the axis of rotation where the modulus is the angle magnitude
        # q = [qx, qy, qz, qw] where qw = cos(theta/2); qx = a1*sin(theta/2),...
        if torch.is_tensor(x):
            qw = x[..., -1]
            theta = 2 * torch.arccos(qw)
            theta = theta.unsqueeze(dim=-1).repeat_interleave(3, dim=-1)
            axis = x[..., :3]/ torch.sin(theta / 2)
            axis[theta==0] = 0  # filter out nans and infs from the division where theta is 0
            # NOTE: should be a unit vector
            x_tr = theta * axis
        else:
            # numpy
            qw = x[..., -1]
            theta = 2 * np.arccos(qw)
            theta = np.expand_dims(theta, axis=-1).repeat(3, axis=-1)
            axis = np.divide(x[..., :3], np.sin(theta/2), out=np.zeros_like(theta), where=theta!=0) # fix that when theta is 0, then axis is (0,0,0) (not defined)
            # NOTE: should be a unit vector
            x_tr = theta * axis
        return x_tr

    @classmethod
    def _tr_inv(cls, x_tr):
        if torch.is_tensor(x_tr):
            theta = torch.norm(x_tr, dim=-1)
            theta = theta.unsqueeze(-1).repeat_interleave(3, axis=-1)
            axis = x_tr / theta
            axis[theta == 0] = 0
            qw = torch.cos(theta[..., 0] * 0.5).unsqueeze(-1)
            qxyz = torch.sin(theta * 0.5) * axis
            x = torch.cat([qxyz, qw], dim=-1)
        else:
            theta = np.linalg.norm(x_tr, axis=-1)
            theta = np.expand_dims(theta, axis=-1).repeat(3, axis=-1)
            axis = np.divide(x_tr, theta, out=np.zeros_like(theta), where=theta!=0)
            qw = np.cos(theta[..., 0:1]*0.5)
            qxyz = np.sin(theta*0.5)*axis
            x = np.concatenate([qxyz, qw], axis=-1)
        return x


class EulerToAxis(object):
    def __init__(self):
        self.quat_to_axis = QuaternionToAxis()

    def euler_sxyz_to_axis_angle(self, euler_sxyz):
        # transform an euler encoded rotation to an axis one with 3 values representing the axis of rotation where the modulus is the angle magnitude
        if euler_sxyz.type == np.ndarray:
            euler_sxyz = torch.from_numpy(euler_sxyz, requires_grad=False)
        euler_reordered = torch.index_select(euler_sxyz, dim=-1, index=torch.LongTensor([2, 1, 0]))
        matrix = batched_trs.euler_angles_to_matrix(euler_reordered, 'ZYX')
        quaternion_wxyz = batched_trs.matrix_to_quaternion(matrix)
        quaternion = torch.index_select(quaternion_wxyz, dim=-1, index=torch.LongTensor([1, 2, 3, 0]))
        axis_angle = torch.from_numpy(self.quat_to_axis._tr(quaternion.detach().numpy()))
        return axis_angle

    def axis_angle_to_euler_sxyz(self, axis_angle):
        matrix = batched_trs.axis_angle_to_matrix(axis_angle)
        euler = batched_trs.matrix_to_euler_angles(matrix, 'ZYX')
        euler_sxyz = torch.index_select(euler, dim=-1, index=torch.LongTensor([2,1,0]))
        return euler_sxyz


# DEBUG
if __name__ == '__main__':
    q2a = QuaternionToAxis()
    simple_sample = {
        'quat': np.array([0, 0, 0, 1]),
    }
    numpy_sample = {
        'quat': np.expand_dims(np.array([0, 0, 0, 1]), axis=0).repeat(5, axis=0),
    }
    tensor_sample = {
        'quat': torch.tensor(np.array([0, 0, 0, 1])).unsqueeze(0).repeat_interleave(5, dim=0),
    }
    simple_sample_tr = q2a(simple_sample)
    numpy_sample_tr = q2a(numpy_sample)
    tensor_sample_tr = q2a(tensor_sample)

    simple_sample_rec = q2a.inverse(simple_sample_tr)
    numpy_sample_rec = q2a.inverse(numpy_sample_tr)
    tensor_sample_rec = q2a.inverse(tensor_sample_tr)


