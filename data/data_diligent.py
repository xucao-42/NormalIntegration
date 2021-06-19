from data.data_class import Data
from scipy.io import loadmat
from matplotlib.pyplot import imread
from utils import camera_to_object, boundary_excluded_mask
import numpy as np
from utils import construct_facets_from_depth_map_mask
import pyvista as pv

def ball_depth_generator(ball_mask):
    pass

class DataDiligent(Data):
    def __init__(self, name, type, exclude_bouday=True):
            if type == "gt":
                self.n = camera_to_object(loadmat("data/diligent/{0}_{1}.mat".format(name, type))["Normal_gt"])
            elif type == "l2":
                self.n = camera_to_object(loadmat("data/diligent/{0}_{1}.mat".format(name, type))["Normal_L2"])
            else:
                self.n = camera_to_object(loadmat("data/diligent/{0}_{1}.mat".format(name, type))["Normal_est"])

            self.mask = imread("data/diligent/{}_mask.png".format(name)).astype(np.bool)
            if self.mask.ndim == 3:
                self.mask = self.mask[..., 0]

            if exclude_bouday:
                self.mask = boundary_excluded_mask(self.mask)
                self.mask = np.logical_and(self.mask, ~np.isclose(np.sum(self.n, axis=-1), 0))
                if name == "goblet":
                    self.mask[115:117, 226] = 0
                    self.mask[171, 220] = 0
                if name == "cat":
                    self.mask[255, 421] = 0
                if name == "buddha":
                    self.mask[104, 335] = 0
            else:
                self.mask = np.logical_and(self.mask, ~np.isclose(np.sum(self.n, axis=-1), 0))
                if name == "goblet":
                    self.mask[[107, 108, 109, 103, 104], [232, 231, 230, 385, 386]] = 0
                if name == "harvest" and type=="CVPR12Ikehata":
                    self.mask[122, 405] = 0

            self.K = np.array([[3759.00543107133, 0, 255.875000000000],
                               [0, 3772.07747101073, 305.875000000000],
                               [0, 0, 1]], dtype=np.float64)
            # self.K = np.array([[3772.07747101073, 0, 305.875000000000],
            #                    [0, 3759.00543107133, 255.875000000000],
            #                    [0, 0, 1]], dtype=np.float64)
            if name != "ball":
                xyz = loadmat("data/diligent/{}PNG_D_XYZ.mat".format(name))["XYZ"]
                self.depth_gt = xyz[..., 2]
                temp = xyz[..., 1].copy()
                xyz[..., 1] = xyz[..., 0].copy()
                xyz[..., 0] = temp
                xyz[..., 0] = -xyz[..., 0]
                self.vertices = xyz[self.mask]
                self.facets = construct_facets_from_depth_map_mask(self.mask)
                self.surf = pv.PolyData(self.vertices, self.facets)
            else:
                self.depth_gt = ball_depth_generator(self.mask)

            self.depth_gt[np.isclose(self.depth_gt, 3000)] = np.nan
            self.projection = "perspective"
            self.fname = name + "_" + type
            self.curl = None
            self.n_vis = (camera_to_object(self.n) + 1)/2
            self.n_vis[~self.mask] = 0

