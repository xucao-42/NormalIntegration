import numpy as np
from utils import *
from scipy.sparse import coo_matrix, hstack
from scipy.sparse.linalg import eigsh
import pyvista as pv


class FivePointPlaneFittingPerspective:
    # working on camera coordinate
    # x
    # |  z
    # | /
    # |/
    # o ---y
    def __init__(self, data, setting):
        self.method = "perspective_five_point_plane_fitting"

        facet_mask = data.mask
        facet_H, facet_W = facet_mask.shape

        facet_idx = np.zeros_like(facet_mask, dtype=np.int)
        facet_idx[facet_mask] = np.arange(np.sum(facet_mask)) + 1  # facet idx begin from 1

        expand_mask = np.pad(facet_mask, 1, "constant", constant_values=0)
        expand_facet_idx = np.pad(facet_idx, 1, "constant", constant_values=0)

        num_plane = np.sum(facet_mask)

        top_neighbor = expand_facet_idx[move_top(expand_mask)].flatten()
        bottom_neighbor = expand_facet_idx[move_bottom(expand_mask)].flatten()
        left_neighbor = expand_facet_idx[move_left(expand_mask)].flatten()
        right_neighbor = expand_facet_idx[move_right(expand_mask)].flatten()
        normal_id_pixel_id = np.hstack((facet_idx[facet_mask][:, None],
                                 top_neighbor[:, None],
                                 bottom_neighbor[:, None],
                                 left_neighbor[:, None],
                                 right_neighbor[:, None]))  # start from 1

        K_inv = np.linalg.inv(data.K)

        yy, xx = np.meshgrid(range(facet_W), range(facet_H))
        xx = np.max(xx) - xx
        xx = xx.astype(np.float)
        yy = yy.astype(np.float)

        v_0 = np.zeros((facet_H, facet_W, 3))
        v_0[..., 0] = xx
        v_0[..., 1] = yy
        v_0[..., 2] = 1
        v_0_f = v_0.reshape(-1, 3).T
        v_0_3d = (K_inv @ v_0_f).T.reshape((facet_H, facet_W, 3))
        vertex_directions = v_0_3d[facet_mask]

        try:
            n_vec = data.n_used[facet_mask]
        except:
            n_vec = data.n[facet_mask]

        # construct the left and the right part of A
        num_eq = np.sum(normal_id_pixel_id != 0)
        row_idx = np.arange(num_eq)
        repeat = np.sum(normal_id_pixel_id != 0, axis=-1)

        A_right_data = np.ones(num_eq)
        A_right_col = np.arange(num_plane)
        A_right_col = np.repeat(A_right_col, repeat)
        A_right = coo_matrix((A_right_data, (row_idx, A_right_col)))

        v_vec = vertex_directions[normal_id_pixel_id - 1]
        A_left_data = np.sum(n_vec[:, None, :] * v_vec, axis=-1).flatten()
        A_left_data = A_left_data[normal_id_pixel_id.flatten() != 0]
        col_idx = (normal_id_pixel_id - 1).flatten()
        col_idx = col_idx[col_idx != -1]
        A_left = coo_matrix((A_left_data, (row_idx, col_idx)))
        A = hstack([A_left, A_right])

        _, x = eigsh(A.T @ A, k=1, sigma=0, which="LM")

        depth = np.squeeze(x[:num_plane])

        self.vertices = vertex_directions * depth[..., None]
        self.facets = construct_facet_for_depth(facet_mask)
        self.surf = pv.PolyData(self.vertices, self.facets)
        depth_map = np.ones_like(facet_mask, dtype=np.float) * np.nan
        depth_map[facet_mask] = depth
        self.depth = depth_map

