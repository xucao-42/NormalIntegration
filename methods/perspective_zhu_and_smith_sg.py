import numpy as np
from scipy.sparse import coo_matrix, diags, vstack, identity
from scipy.sparse.linalg import eigsh
from utils import construct_facets_from_depth_map_mask, map_depth_map_to_point_clouds
import pyvista as pv
from scipy.spatial import KDTree
import time

class PerspectiveZhuSG:
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
    def __init__(self, data, setting):
        self.method_name = "perspective_zhu_and_smith_sg_order_{0}_neighbor_{1}_lambda_smooth_{2}"\
            .format(setting.polynomial_order, setting.num_neighbor, setting.lambda_smooth).replace(".", "_")
        method_start = time.time()

        num_pixel = np.sum(data.mask)
        H, W = data.mask.shape

        vv, uu = np.meshgrid(range(W), range(H))
        uu = np.flip(uu, axis=0)

        ox = data.K[0, 2]
        oy = data.K[1, 2]
        fx = data.K[0, 0]
        fy = data.K[1, 1]

        u = (uu - ox)[data.mask]
        v = (vv - oy)[data.mask]

        # search for k nearest neighbour pixels for each pixel in region of integration
        try:
            self.neighbor_pixel_idx = data.neighbour_idx
        except:
            pixel_coordinates = np.concatenate([u[..., np.newaxis],
                                             v[..., np.newaxis]], axis=-1)

            _, neighbor_pixel_idx = KDTree(pixel_coordinates).query(pixel_coordinates, k=setting.num_neighbor)
            self.neighbor_pixel_idx = np.sort(neighbor_pixel_idx, axis=-1)

        # retrieve neighbour points' pixel coordinates
        center_u = u
        center_v = v
        neighbour_u = u[neighbor_pixel_idx]
        neighbour_v = v[neighbor_pixel_idx]
        poly_list = []
        order_list = []
        for i in range(setting.polynomial_order + 1):
            for j in range(setting.polynomial_order + 1 - i):
                order_list.append((i, j))
                # each row in Eq. (16) of "Least squares surface reconstruction on arbitrary domains."
                c_i = ((neighbour_u - center_u[..., np.newaxis]) ** i) * ((neighbour_v - center_v[..., np.newaxis]) ** j)
                poly_list.append(c_i[..., np.newaxis])
        C = np.concatenate(poly_list, axis=-1)
        C_pinv = np.linalg.pinv(C)  # num_pixels x num_polynomials x num_neighbour
        a00 = C_pinv[:, order_list.index((0, 0)), :].flatten()  # num_pixels x num_neighbour. smoothness term

        # construct Du and Dv based on SG filters
        a10 = C_pinv[:, order_list.index((1, 0)), :].flatten()
        a01 = C_pinv[:, order_list.index((0, 1)), :].flatten()
        row_idx = np.arange(num_pixel)
        row_idx = np.repeat(row_idx, setting.num_neighbor)
        col_idx = neighbor_pixel_idx.flatten()
        Du = coo_matrix((a10, (row_idx, col_idx)), shape=(num_pixel, num_pixel))
        Dv = coo_matrix((a01, (row_idx, col_idx)), shape=(num_pixel, num_pixel))

        # smoothness penalty in Sec.4
        try:
            S = data.S
        except:
            S = coo_matrix((a00, (row_idx, col_idx)), shape=(num_pixel, num_pixel))
        self.S = S

        # Eq. (11) in "Least squares surface reconstruction on arbitrary domains."
        U = diags(u)
        V = diags(v)

        nx = data.n[data.mask, 0]
        ny = data.n[data.mask, 1]
        nz = data.n[data.mask, 2]

        N = vstack([diags(nx),
                    diags(ny),
                    diags(nz)]).T

        Tx = vstack([(U @ Du + identity(num_pixel)) / fx,
                     V @ Du / fy,
                     Du])
        Ty = vstack([U @ Dv / fx,
                     (V @ Dv + identity(num_pixel)) / fy,
                     Dv])

        # Eq. (10) in "Least squares surface reconstruction on arbitrary domains."
        A = vstack([N @ Tx,
                    N @ Ty,
                    setting.lambda_smooth * (S - identity(num_pixel))])

        solver_start = time.time()

        _, z = eigsh(A.T @ A, k=1, sigma=0, which="LM")

        solver_end = time.time()
        self.solver_runtime = solver_end - solver_start

        method_end = time.time()
        self.total_runtime = method_end - method_start

        self.depth_map = np.ones_like(data.mask, dtype=np.float) * np.nan
        self.depth_map[data.mask] = np.squeeze(z)

        # construct a mesh from the depth map
        self.facets = construct_facets_from_depth_map_mask(data.mask)
        self.vertices = map_depth_map_to_point_clouds(self.depth_map, data.mask, data.K)
        self.surface = pv.PolyData(self.vertices, self.facets)
