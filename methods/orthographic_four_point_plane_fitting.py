import numpy as np
from utils import *
from scipy.sparse import coo_matrix, hstack
import pyvista as pv
from scipy.sparse.linalg import lsqr


class FourPointOrthographic:
    # x
    # |  z
    # | /
    # |/
    # o ---y
    def __init__(self, data, setting):
        self.method_name = "orthographic_four_point_plane_fitting"

        facet_mask = data.mask
        facet_H, facet_W = facet_mask.shape

        facet_idx = np.zeros_like(facet_mask, dtype=np.int)
        facet_idx[facet_mask] = np.arange(np.sum(facet_mask)) + 1  # facet idx begin from 1

        top_left_mask = np.pad(facet_mask, ((0, 1), (0, 1)), "constant", constant_values=0)
        top_right_mask = np.pad(facet_mask, ((0, 1), (1, 0)), "constant", constant_values=0)
        bottom_left_mask = np.pad(facet_mask, ((1, 0), (0, 1)), "constant", constant_values=0)
        bottom_right_mask = np.pad(facet_mask, ((1, 0), (1, 0)), "constant", constant_values=0)

        vertex_mask = np.logical_or.reduce((top_right_mask, top_left_mask, bottom_right_mask, bottom_left_mask))
        vertex_idx = np.zeros((facet_H + 1, facet_W + 1), dtype=np.int)
        vertex_idx[vertex_mask] = np.arange(np.sum(vertex_mask)) + 1  # vertex idx begin from 1

        num_facet = np.sum(facet_mask)
        num_vertex = np.sum(vertex_mask)

        top_left_vertex = vertex_idx[top_left_mask].flatten()
        top_right_vertex = vertex_idx[top_right_mask].flatten()
        bottom_left_vertex = vertex_idx[bottom_left_mask].flatten()
        bottom_right_vertex = vertex_idx[bottom_right_mask].flatten()
        facet_id_vertice_id = np.hstack((top_left_vertex[:, None],
                                         bottom_left_vertex[:, None],
                                         bottom_right_vertex[:, None],
                                         top_right_vertex[:, None]))  # start from 1

        yy, xx = np.meshgrid(range(facet_W + 1), range(facet_H + 1))
        xx = np.max(xx) - xx
        xx = xx.astype(np.float)
        yy = yy.astype(np.float)
        xx -= 0.5
        yy -= 0.5

        xx *= data.step_size
        yy *= data.step_size

        v_0 = np.zeros((facet_H + 1, facet_W + 1, 3))
        v_0[..., 0] = xx
        v_0[..., 1] = yy

        # center directions are used for extrcting depth values at Omega_n
        center_yy, center_xx = np.meshgrid(range(facet_W), range(facet_H))
        center_xx = np.max(center_xx) - center_xx
        center_xx = center_xx.astype(np.float)[facet_mask]
        center_yy = center_yy.astype(np.float)[facet_mask]
        center_xx *= data.step_size
        center_yy *= data.step_size

        center_points = np.zeros((facet_H, facet_W, 3))
        center_points[facet_mask, 0] = center_xx
        center_points[facet_mask, 1] = center_yy

        # construct the left and the right part of A
        try:
            n_vec = data.n_used[facet_mask]
        except:
            n_vec = data.n[facet_mask]

        num_eq = np.sum(facet_id_vertice_id != 0)
        row_idx = np.arange(num_eq)

        data_ = n_vec[:, 2]
        data_ = np.repeat(data_, 4)
        col_idx = (facet_id_vertice_id - 1).flatten()
        A_left = coo_matrix((data_, (row_idx, col_idx)))

        A_right_data = np.ones(num_eq)
        A_right_col = np.arange(num_facet)
        A_right_col = np.repeat(A_right_col, 4)
        A_right = coo_matrix((A_right_data, (row_idx, A_right_col)))

        A = hstack([A_left, A_right])

        u_vec = xx[vertex_mask]
        v_vec = yy[vertex_mask]
        n1 = n_vec[:, 0]
        n2 = n_vec[:, 1]
        n1 = np.repeat(n1, 4)
        n2 = np.repeat(n2, 4)

        u_vec = u_vec[facet_id_vertice_id - 1].flatten()
        v_vec = v_vec[facet_id_vertice_id - 1].flatten()
        b = - u_vec * n1 - v_vec * n2

        z = lsqr(A, b)[0]
        self.res = A @ z - b

        vertex_depth = np.squeeze(z[:num_vertex])
        vertex_facets = construct_facet_for_depth(vertex_mask)
        v_0[vertex_mask, 2] = vertex_depth
        vertex = v_0[vertex_mask]
        self.vertex_surf = pv.PolyData(vertex, vertex_facets)

        plane_displacement = np.squeeze(z[num_vertex:])
        center_depth = (- plane_displacement - center_xx * n_vec[:, 0] - center_yy * n_vec[:, 1]) / n_vec[:, 2]
        center_points[facet_mask, 2] = center_depth


        self.depth_facets = construct_facet_for_depth(facet_mask)
        self.surf = pv.PolyData(center_points[facet_mask], self.depth_facets)

        depth_map = np.ones_like(facet_mask, dtype=np.float) * np.nan
        depth_map[facet_mask] = center_depth
        self.depth = depth_map

