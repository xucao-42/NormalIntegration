from scipy.sparse import diags
import numpy as np
from utils import construct_facet_for_depth
from scipy.linalg import solve_sylvester
import pyvista as pv

def generate_discrete_diff(H):
    diag0 = np.array([-3] + [0] * (H-2) + [3], np.float)
    diag__1 = np.array([-1] * (H-2) + [-4], np.float)
    diag_1 = np.array([4] + [1] * (H-2), np.float)
    diag__2 = np.array([0] * (H-3) + [1], np.float)
    diag_2 = np.array([-1] + [0] * (H-3), np.float)

    D = diags([diag__2, diag__1, diag0, diag_1, diag_2], [-2, -1, 0, 1, 2], shape=(H, H))
    return D



class HarkerOrthographic:
    def __init__(self, data, setting):
        self.method_name = "orthographic_harker"
        mask = data.mask
        H, W = mask.shape
        yy, xx = np.meshgrid(range(W), range(H))
        xx = np.max(xx) - xx
        xx = xx.astype(np.float)
        yy = yy.astype(np.float)
        xx *= data.step_size
        yy *= data.step_size

        zy_hat = - data.n[..., 0] / data.n[..., 2]
        zx_hat = - data.n[..., 1] / data.n[..., 2]

        Dy = - generate_discrete_diff(H) / (2 * data.step_size)
        Dx = generate_discrete_diff(W) / (2 * data.step_size)

        A = Dy.T @ Dy
        B = Dx.T @ Dx
        C = Dy.T @ zy_hat + zx_hat @ Dx

        z_map = solve_sylvester(A.toarray(), B.toarray(), C)
        self.depth = -z_map

        # construct a mesh from the depth map
        self.facets = construct_facet_for_depth(mask)
        v_0 = np.zeros((H, W, 3))
        v_0[..., 0] = xx
        v_0[..., 1] = yy
        v_0[..., 2] = - z_map
        self.vertices = v_0[mask].reshape(-1, 3)
        self.surf = pv.PolyData(self.vertices, self.facets)

