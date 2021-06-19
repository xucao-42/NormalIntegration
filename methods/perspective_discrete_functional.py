import numpy as np
from scipy.sparse.linalg import lsqr
from utils import construct_facets_from_depth_map_mask, map_depth_map_to_point_clouds
import pyvista as pv
from orthographic_discrete_functional import generate_dx_dy_wb
from scipy.sparse import vstack
import time

class PerspectiveDiscreteFunctional:
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
        self.method_name = "perspective_discrete_functional"
        method_start = time.time()

        H, W = data.mask.shape
        yy, xx = np.meshgrid(range(W), range(H))
        xx = np.flip(xx, axis=0)

        ox = data.K[0, 2]
        oy = data.K[1, 2]
        fx = data.K[0, 0]
        fy = data.K[1, 1]

        uu = xx - ox
        vv = yy - oy

        n1 = data.n[..., 0]
        n2 = data.n[..., 1]
        n3 = data.n[..., 2]

        # Eq. (17) in "Normal Integration: A Survey."
        p_tilde = - n1 / (uu * n1 + vv * n2 + fx * n3)
        q_tilde = - n2 / (uu * n1 + vv * n2 + fy * n3)

        b = np.concatenate((p_tilde[data.mask], q_tilde[data.mask]))
        Du, Dv = generate_dx_dy_wb(data.mask)
        A = vstack((Du, Dv))

        solver_start = time.time()

        z_tilde = lsqr(A, b)[0]
        # Eq. (13) in "Normal Integration: A Survey."
        z = np.exp(z_tilde)

        solver_end = time.time()
        self.solver_runtime = solver_end - solver_start

        method_end = time.time()
        self.total_runtime = method_end - method_start

        self.depth_map = np.ones_like(data.mask, dtype=np.float) * np.nan
        self.depth_map[data.mask] = z

        # construct a mesh from the depth map
        self.facets = construct_facets_from_depth_map_mask(data.mask)
        self.vertices = map_depth_map_to_point_clouds(self.depth_map, data.mask, data.K)
        self.surface = pv.PolyData(self.vertices, self.facets)





