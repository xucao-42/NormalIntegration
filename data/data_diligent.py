from data.data_class import Data
from scipy.io import loadmat
from matplotlib.pyplot import imread
from utils import camera_to_object, boundary_excluded_mask
import numpy as np
from utils import construct_facets_from_depth_map_mask
import pyvista as pv
import os


class DataDiligent(Data):
    def __init__(self, surface_name, surface_method, exclude_bouday=True):
            if surface_method == "gt":
                self.n = camera_to_object(loadmat(os.path.join("data", "DiLiGenT", "{0}_{1}.mat".format(surface_name, surface_method)))["Normal_gt"])
            elif surface_method == "l2":
                self.n = camera_to_object(loadmat(os.path.join("data", "DiLiGenT", "{0}_{1}.mat".format(surface_name, surface_method)))["Normal_L2"])
            else:
                self.n = camera_to_object(loadmat(os.path.join("data", "DiLiGenT", "{0}_{1}.mat".format(surface_name, surface_method)))["Normal_est"])

            self.mask = imread(os.path.join("data", "DiLiGenT", "{}_mask.png".format(surface_name))).astype(np.bool)
            if self.mask.ndim == 3:
                self.mask = self.mask[..., 0]

            if exclude_bouday:
                self.mask = boundary_excluded_mask(self.mask)
                self.mask = np.logical_and(self.mask, ~np.isclose(np.sum(self.n, axis=-1), 0))
                if surface_name == "goblet":
                    self.mask[115:117, 226] = 0
                    self.mask[171, 220] = 0
            else:
                self.mask = np.logical_and(self.mask, ~np.isclose(np.sum(self.n, axis=-1), 0))
                if surface_name == "goblet":
                    self.mask[[107, 108, 109, 103, 104], [232, 231, 230, 385, 386]] = 0
                if surface_name == "harvest":
                    self.mask[122, 405] = 0

            self.K = np.array([[3759.00543107133, 0, 255.875000000000],
                               [0, 3772.07747101073, 305.875000000000],
                               [0, 0, 1]], dtype=np.float64)
            # self.K = np.array([[3772.07747101073, 0, 305.875000000000],
            #                    [0, 3759.00543107133, 255.875000000000],
            #                    [0, 0, 1]], dtype=np.float64)
            if surface_name != "ball":
                xyz = loadmat(os.path.join("data", "DiLiGenT", "{}PNG_D_XYZ.mat".format(surface_name)))["XYZ"]
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
            self.fname = surface_name + "_" + surface_method
            self.curl = None
            self.n_vis = (camera_to_object(self.n) + 1)/2
            self.n_vis[~self.mask] = 1
