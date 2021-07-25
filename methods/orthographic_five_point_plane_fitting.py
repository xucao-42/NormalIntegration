from utils import *
from scipy.sparse import coo_matrix, hstack
import pyvista as pv
from scipy.sparse.linalg import lsqr
import pyamg
import time

class OrthographicFivePoint:
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
        self.method_name = "orthographic_five_point_plane_fitting"
        method_start = time.time()
        H, W = data.mask.shape
        vv, uu = np.meshgrid(range(W), range(H))
        uu = np.flip(uu, axis=0)

        # pixel_coordinates = np.stack((uu[data.mask], vv[data.mask]), axis=-1)
        # from sklearn.neighbors import KDTree
        # tree = KDTree(pixel_coordinates)
        # neighbor_pixel_ids = tree.query_radius(pixel_coordinates, r=1 + 1e-7)

        # For each pixel, search for its neighbor pixel indices and store them as an item in the list neighbor_pixel_ids
        # including itself's index
        pixel_idx = np.zeros_like(data.mask, dtype=np.int)
        pixel_idx[data.mask] = np.arange(np.sum(data.mask)) + 1
        # pixel indices starts from 1 to ensure all pixels with 0 indices are background pixels

        expand_mask = np.pad(data.mask, 1, "constant", constant_values=0)
        expand_pixel_idx = np.pad(pixel_idx, 1, "constant", constant_values=0)

        top_neighbor = expand_pixel_idx[move_top(expand_mask)]
        bottom_neighbor = expand_pixel_idx[move_bottom(expand_mask)]
        left_neighbor = expand_pixel_idx[move_left(expand_mask)]
        right_neighbor = expand_pixel_idx[move_right(expand_mask)]

        neighbor_pixel_ids = np.stack((pixel_idx[data.mask], top_neighbor, bottom_neighbor, left_neighbor, right_neighbor), axis=-1)

        neighbor_pixel_ids = [i[i!=0] - 1 for i in neighbor_pixel_ids]
        # pixels in the background are filtered out (i!=0),
        # and pixel indices start from 0 now to construct the sparse coefficient matrix

        # construct the system matrix A based on the list of neighbor pixel indices
        num_neighbor_list = [len(i) for i in neighbor_pixel_ids]  # number of neighbour pixels for each pixel
        num_plane_equations = sum(num_neighbor_list)

        nz = data.n[data.mask, 2]
        nz = np.repeat(nz, num_neighbor_list)

        col_idx = np.concatenate(neighbor_pixel_ids)
        row_idx = np.arange(num_plane_equations)
        A_left = coo_matrix((nz, (row_idx, col_idx)))

        all_ones = np.ones(num_plane_equations)
        num_normals = np.sum(data.mask)
        A_right_col = np.arange(num_normals)
        A_right_col = np.repeat(A_right_col, num_neighbor_list)
        A_right = coo_matrix((all_ones, (row_idx, A_right_col)))

        A = hstack([A_left, A_right])

        # the stacking of right-hand side of Eq. (12) at all pixels
        # "Normal Integration via Inverse Plane Fitting with Minimum Point-to-Plane Distance"
        u = uu[data.mask] * data.step_size
        v = vv[data.mask] * data.step_size
        u = u[col_idx]
        v = v[col_idx]

        nx = data.n[data.mask, 0]
        ny = data.n[data.mask, 1]
        nx = np.repeat(nx, num_neighbor_list)
        ny = np.repeat(ny, num_neighbor_list)

        b = - u * nx - v * ny

        solver_start = time.time()
        z = lsqr(A, b, atol=1e-17, btol=1e-17)[0]
        # z = pyamg.solve(A.T @ A, A.T @ b, tol=1e-17)
        solver_end = time.time()

        self.solver_runtime = solver_end - solver_start

        method_end = time.time()
        self.total_runtime = method_end - method_start

        self.residual = A @ z - b

        self.depth_map = np.ones_like(data.mask, dtype=np.float) * np.nan
        self.depth_map[data.mask] = z[:num_normals]

        # create a mesh model from the depth map
        self.facets = construct_facets_from_depth_map_mask(data.mask)
        self.vertices = construct_vertices_from_depth_map_and_mask(data.mask, self.depth_map, data.step_size)
        self.surface = pv.PolyData(self.vertices, self.facets)


