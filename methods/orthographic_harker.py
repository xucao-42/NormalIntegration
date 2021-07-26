import sys
sys.path.append("..")
sys.path.append(".")

from scipy.sparse import diags
import numpy as np
from utils import construct_facets_from_depth_map_mask, construct_vertices_from_depth_map_and_mask
from scipy.linalg import solve_sylvester
import pyvista as pv
import time


def generate_discrete_diff(H):
    # Eq. (11) in "Regularized Reconstruction of a Surface from its Measured Gradient Field."
    diag0 = np.array([-3] + [0] * (H-2) + [3], np.float)
    diag__1 = np.array([-1] * (H-2) + [-4], np.float)
    diag_1 = np.array([4] + [1] * (H-2), np.float)
    diag__2 = np.array([0] * (H-3) + [1], np.float)
    diag_2 = np.array([-1] + [0] * (H-3), np.float)

    D = diags([diag__2, diag__1, diag0, diag_1, diag_2], [-2, -1, 0, 1, 2], shape=(H, H))
    return D


class OrthographicHarker:
    # camera coordinates
    # x
    # |  z
    # | /
    # |/
    # o ---y
    # pixel coordinates
    # u
    # |
    # |
    # |
    # o ---v
    def __init__(self, data):
        self.method_name = "orthographic_harker"
        print("running {}...".format(self.method_name))
        method_start = time.time()

        H, W = data.mask.shape
        zy_hat = - data.n[..., 0] / data.n[..., 2]
        zx_hat = - data.n[..., 1] / data.n[..., 2]

        Dy = - generate_discrete_diff(H) / (2 * data.step_size)
        Dx = generate_discrete_diff(W) / (2 * data.step_size)

        # Eq. (21) in paper "Regularized Reconstruction of a Surface from its Measured Gradient Field."
        A = Dy.T @ Dy
        B = Dx.T @ Dx
        C = Dy.T @ zy_hat + zx_hat @ Dx

        A = A.toarray()
        B = B.toarray()

        solver_start = time.time()
        self.depth_map = - solve_sylvester(A, B, C)
        solver_end = time.time()

        self.solver_running_time = solver_end - solver_start
        method_end = time.time()
        self.total_runtime = method_end - method_start

        # construct a mesh from the depth map
        self.facets = construct_facets_from_depth_map_mask(data.mask)
        self.vertices = construct_vertices_from_depth_map_and_mask(data.mask, self.depth_map, data.step_size)
        self.surface = pv.PolyData(self.vertices, self.facets)

