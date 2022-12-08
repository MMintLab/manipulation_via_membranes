import numpy as np
import torch
import matplotlib.pyplot as plt


class FittedGaussianPoseLoss(torch.nn.Module):

    def __init__(self, img_shape, n_points = 20, device=None, criterion=None):
        super().__init__()
        self.device = device
        self.criterion = criterion
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if criterion is None:
            self.criterion = torch.nn.MSELoss()
        self.img_shape = img_shape
        self.n_points = n_points


    def forward(self, gaussian_pred, gaussian_gth):
        """
        Assumptions:
        """
        # print('First gth output')
        # self.print_twoD_Gaussian(xy, gaussian_gth[1][0], gaussian_gth[1][1], gaussian_gth[1][2], gaussian_gth[1][3], gaussian_gth[1][4], gaussian_gth[1][5], gaussian_gth[1][6])
        batch_size = gaussian_pred.shape[0]
        m_gth_1, n_gth_1 = self.twoD_Gaussian_points(gaussian_gth[:,:7], self.n_points, count=True)
        m_pred_1, n_pred_1 = self.twoD_Gaussian_points(gaussian_pred[:,:7], self.n_points, count=True)
        gth_1_inside = (n_gth_1 > self.n_points/4).type(torch.int)
        gth_1_fact = torch.tile(gth_1_inside, (self.n_points, 2, 1)).permute((2,0,1))
        pred_1_inside = (n_pred_1 - n_gth_1 < -self.n_points/2).type(torch.int)
        pred_1_fact = torch.tile(pred_1_inside, (self.n_points, 2, 1)).permute((2,0,1))

        m_gth_2, n_gth_2 = self.twoD_Gaussian_points(gaussian_gth[:,7:], self.n_points, count=True)
        m_pred_2, n_pred_2 = self.twoD_Gaussian_points(gaussian_pred[:,7:], self.n_points, count=True)
        gth_2_inside = (n_gth_2 > self.n_points/2).type(torch.int)
        gth_2_fact = torch.tile(gth_2_inside, (self.n_points, 2, 1)).permute((2,0,1))
        pred_2_inside = (n_pred_2 - n_gth_2 < -self.n_points/2).type(torch.int)
        pred_2_fact = torch.tile(pred_2_inside*100+1, (self.n_points, 2, 1)).permute((2,0,1))
        loss_1 = self.criterion(torch.norm((m_pred_1-m_gth_1)*gth_1_fact*pred_1_fact, dim=-1), torch.norm(m_gth_1-m_gth_1, dim=-1))
        loss_2 = self.criterion(torch.norm((m_pred_2-m_gth_2)*gth_2_fact*pred_2_fact, dim=-1), torch.norm(m_gth_2-m_gth_2, dim=-1))
        loss = (loss_1 + loss_2)/(torch.numel(m_pred_1))
        return loss


    def print_twoD_Gaussian(self, xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        xo = float(xo)
        yo = float(yo)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = offset + amplitude*np.exp( - (a*((xy[0]-xo)**2) + 2*b*(xy[0]-xo)*(xy[1]-yo) + c*((xy[1]-yo)**2)))
        plt.imshow(g)
        plt.show()
        return g.ravel()

    def _sample_circle(self, n_points, radius):
        theta = np.linspace(0, 2*np.pi, n_points)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points = np.stack((x,y)).T
        return points

    def _gaussian_transformation(self, gaussian, points, count=False):
        batch_size = gaussian.shape[0]
        points_t = torch.tile(torch.tensor(points.T),(batch_size,1,1)).type(torch.float).to(self.device)
        xo = torch.tile(gaussian[:,1], (points_t.shape[2], 1)).T
        yo = torch.tile(gaussian[:,2], (points_t.shape[2], 1)).T
        sigma_x = gaussian[:,3]
        sigma_y = gaussian[:,4]
        theta = gaussian[:,5].detach()
        Sigma = torch.zeros((batch_size, 2, 2)).to(self.device)
        Sigma[:, 0, 0] = sigma_x ** 2
        Sigma[:, 0, 1] = Sigma[:, 1, 0] = theta * sigma_x * sigma_y
        Sigma[:, 1, 1] = sigma_y ** 2
        U, S, V_t = torch.linalg.svd(Sigma)
        S_sqrt = torch.zeros_like(Sigma).to(self.device)
        S_sqrt[:, 0, 0] = torch.sqrt(S[:, 0])
        S_sqrt[:, 1, 1] = torch.sqrt(S[:, 1])
        transformed_points = 2 * (U @ S_sqrt @ V_t @ points_t)
        transformed_points[:, 0, :] += xo
        transformed_points[:, 1, :] += yo
        transformed_points = transformed_points.permute((0,2,1))
        transformed_points = torch.flip(transformed_points, [-1])
        transformed_points_inside = torch.logical_and(transformed_points[:,:,0] > 0, torch.logical_and(transformed_points[:,:,0] < self.img_shape[0],
                                    torch.logical_and(transformed_points[:,:,1] > 0, transformed_points[:,:,1] < self.img_shape[1])))
        n_points_inside = torch.count_nonzero(transformed_points_inside, dim=1)
        if count:
            return transformed_points, n_points_inside
        return transformed_points 

    def twoD_Gaussian_points(self, gaussian, n_points, count=False):
        points = self._sample_circle(n_points, 0.75)
        ellipsoid_points = self._gaussian_transformation(gaussian, points, count)
        return ellipsoid_points

    # def twoD_Gaussian_points(self, gaussian, n_points): # xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset, n_points, value):
    #     # import pdb; pdb.set_trace()
    #     batch_size = gaussian.shape[0]
    #     # Create grid
    #     x = np.linspace(0, self.img_shape[1]-1, self.img_shape[1])
    #     y = np.linspace(0, self.img_shape[0]-1, self.img_shape[0])
    #     x, y = torch.tensor(np.meshgrid(x, y))
    #     x = torch.tile(x,(batch_size,1,1))
    #     y = torch.tile(y,(batch_size,1,1))
    #     # Load parameters
    #     amplitude = gaussian[:,0]
    #     xo = gaussian[:,1]
    #     yo = gaussian[:,2]
    #     sigma_x = gaussian[:,3]
    #     sigma_y = gaussian[:,4]
    #     theta = gaussian[:,5]
    #     offset = gaussian[:,6]

    #     amplitude = (torch.tile(amplitude, (self.img_shape[0],self.img_shape[1],1))).permute(2,0,1)
    #     xo = (torch.tile(xo, (self.img_shape[0],self.img_shape[1],1))).permute(2,0,1)
    #     yo = (torch.tile(yo, (self.img_shape[0],self.img_shape[1],1))).permute(2,0,1)
    #     sigma_x = (torch.tile(sigma_x, (self.img_shape[0],self.img_shape[1],1))).permute(2,0,1)
    #     sigma_y = (torch.tile(sigma_y, (self.img_shape[0],self.img_shape[1],1))).permute(2,0,1)
    #     theta = (torch.tile(theta, (self.img_shape[0],self.img_shape[1],1))).permute(2,0,1)
    #     offset = (torch.tile(offset, (self.img_shape[0],self.img_shape[1],1))).permute(2,0,1)

    #     A = (torch.cos(theta)**2)/(2*sigma_x**2) + (torch.sin(theta)**2)/(2*sigma_y**2)
    #     B = -(torch.sin(2*theta))/(4*sigma_x**2) + (torch.sin(2*theta))/(4*sigma_y**2)
    #     C = (torch.sin(theta)**2)/(2*sigma_x**2) + (torch.cos(theta)**2)/(2*sigma_y**2)
    #     # Calculate gaussian
    #     g = amplitude*torch.exp( - (A*((x-xo)**2) + 2*B*(x-xo)*(y-yo) + C*((y-yo)**2))).detach()
    #     # Decide level where to take the contour
    #     coef = 0.5
    #     level = (amplitude[:,0,0]*torch.exp( - (A[:,0,0]*((sigma_x[:,0,0]*coef)**2) + 
    #             2*B[:,0,0]*(sigma_x[:,0,0]*coef)*(sigma_y[:,0,0]*coef) + C[:,0,0]*((sigma_y[:,0,0]*coef)**2)))).detach()
        
    #     # Calculate points of the ellipsoid
    #     points = []
    #     for i, _ in enumerate(g):
    #         cs = plt.contour(x[i], y[i], g[i], [level[i]])
    #         vertices = np.zeros((n_points, 2))
    #         if len(cs.collections[0].get_paths()) > 0:
    #             p = cs.collections[0].get_paths()[0]
    #             v = p.vertices
    #             n = len(v)
    #             m = np.ceil(n/n_points).astype(int)
    #             vertices = v[::m]
    #             point_ind = np.random.randint(v.shape[0], size=n_points-len(vertices)).tolist()
    #             vertices = np.concatenate((vertices, v[point_ind]))
    #             vertices = np.flip(vertices)
    #         points.append(vertices)
    #     return torch.tensor(points)
