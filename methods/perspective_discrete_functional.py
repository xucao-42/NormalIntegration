import numpy as np
from scipy.sparse.linalg import lsqr
from utils import construct_facet_for_depth, hide_all_plot
import pyvista as pv
import os
from scipy.sparse import coo_matrix, vstack


def generate_dx_dy_wb(normal_mask, h):
    all_depth_idx = np.zeros_like(normal_mask, dtype=np.int)
    all_depth_idx[normal_mask] = np.arange(np.sum(normal_mask))
    num_depth = np.sum(normal_mask)

    move_left_mask = np.pad(normal_mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]
    move_right_mask = np.pad(normal_mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]
    move_top_mask = np.pad(normal_mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]
    move_bottom_mask = np.pad(normal_mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]

    has_left_and_right_mask = np.logical_and.reduce((move_left_mask, move_right_mask, normal_mask))
    has_only_left_mask = np.logical_and(np.logical_xor(move_left_mask, normal_mask), normal_mask)
    has_only_right_mask = np.logical_and(np.logical_xor(move_right_mask, normal_mask), normal_mask)

    has_left_and_right_mask_left = np.pad(has_left_and_right_mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]
    has_left_and_right_mask_right = np.pad(has_left_and_right_mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]

    has_only_left_mask_left = np.pad(has_only_left_mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]
    has_only_right_mask_right = np.pad(has_only_right_mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]

    row_idx = np.concatenate([all_depth_idx[has_only_left_mask].flatten(),
                              all_depth_idx[has_only_right_mask].flatten(),
                              all_depth_idx[has_left_and_right_mask].flatten(),
                              all_depth_idx[has_only_left_mask].flatten(),
                              all_depth_idx[has_only_right_mask].flatten(),
                              all_depth_idx[has_left_and_right_mask].flatten()])

    col_idx = np.concatenate([all_depth_idx[has_only_left_mask_left].flatten(),
                              all_depth_idx[has_only_right_mask_right].flatten(),
                              all_depth_idx[has_left_and_right_mask_left].flatten(),
                              all_depth_idx[has_only_left_mask].flatten(),
                              all_depth_idx[has_only_right_mask].flatten(),
                              all_depth_idx[has_left_and_right_mask_right].flatten()])
    data_term = [-1] * np.sum(has_only_left_mask) + [1] * np.sum(has_only_right_mask) + [-0.5] * np.sum(has_left_and_right_mask) \
                + [1] * np.sum(has_only_left_mask_left) + [-1] * np.sum(has_only_right_mask_right) + [0.5] * np.sum(has_left_and_right_mask)
    D_horizontal = coo_matrix((data_term, (row_idx, col_idx)), shape=(num_depth, num_depth)) / h

    has_bottom_and_top_mask = np.logical_and.reduce((move_bottom_mask, move_top_mask, normal_mask))
    has_only_bottom_mask = np.logical_and(np.logical_xor(move_bottom_mask, normal_mask), normal_mask)
    has_only_top_mask = np.logical_and(np.logical_xor(move_top_mask, normal_mask), normal_mask)

    has_bottom_and_top_mask_bottom = np.pad(has_bottom_and_top_mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]
    has_bottom_and_top_mask_top = np.pad(has_bottom_and_top_mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:,:]

    has_only_bottom_mask_bottom = np.pad(has_only_bottom_mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]
    has_only_top_mask_top = np.pad(has_only_top_mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]

    row_idx = np.concatenate([all_depth_idx[has_only_bottom_mask].flatten(),
                              all_depth_idx[has_only_top_mask].flatten(),
                              all_depth_idx[has_bottom_and_top_mask].flatten(),
                              all_depth_idx[has_only_bottom_mask].flatten(),
                              all_depth_idx[has_only_top_mask].flatten(),
                              all_depth_idx[has_bottom_and_top_mask].flatten()])

    col_idx = np.concatenate([all_depth_idx[has_only_bottom_mask_bottom].flatten(),
                              all_depth_idx[has_only_top_mask_top].flatten(),
                              all_depth_idx[has_bottom_and_top_mask_bottom].flatten(),
                              all_depth_idx[has_only_bottom_mask].flatten(),
                              all_depth_idx[has_only_top_mask].flatten(),
                              all_depth_idx[has_bottom_and_top_mask_top].flatten()])
    data_term = [-1] * np.sum(has_only_bottom_mask) + [1] * np.sum(has_only_top_mask) + [-0.5] * np.sum(
        has_bottom_and_top_mask) \
                + [1] * np.sum(has_only_bottom_mask_bottom) + [-1] * np.sum(has_only_top_mask_top) + [0.5] * np.sum(
        has_bottom_and_top_mask)
    D_vertical = coo_matrix((data_term, (row_idx, col_idx)), shape=(num_depth, num_depth)) / h


    A = vstack((D_vertical, D_horizontal))
    return A


class DiscreteFunctionalPerspective:
    # working on camera coordinate
    # x
    # |  z
    # | /
    # |/
    # o ---y
    def __init__(self, data, setting):
        self.method = "perspective_discrete_functional"
        mask = data.mask
        H, W = mask.shape
        yy, xx = np.meshgrid(range(W), range(H))
        xx = np.max(xx) - xx
        xx = xx.astype(np.float)
        yy = yy.astype(np.float)

        ox = data.K[0, 2]
        oy = data.K[1, 2]
        fx = data.K[0, 0]
        fy = data.K[1, 1]

        u_prime = xx - ox
        v_prime = yy - oy

        try:
            n1 = data.n_used[..., 0]
            n2 = data.n_used[..., 1]
            n3 = data.n_used[..., 2]
        except:
            n1 = data.n[..., 0]
            n2 = data.n[..., 1]
            n3 = data.n[..., 2]

        p_tilde = - n1 / (u_prime * n1 + v_prime * n2 + fx * n3)
        q_tilde = - n2 / (u_prime * n1 + v_prime * n2 + fy * n3)

        b = np.concatenate((p_tilde[mask], q_tilde[mask]))
        A = generate_dx_dy_wb(mask, 1)

        z_tilde = lsqr(A, b)[0]
        z = np.exp(z_tilde)

        z_map = np.zeros_like(mask, dtype=np.float)
        z_map[mask] = z
        z_map[~mask] = np.nan
        self.depth = z_map

        # construct a mesh from the depth map
        self.facets = construct_facet_for_depth(mask)
        K_1 = np.linalg.inv(data.K)
        v_0 = np.zeros((H, W, 3))
        v_0[..., 0] = xx
        v_0[..., 1] = yy
        v_0[..., 2] = 1
        v_0_f = v_0[mask].reshape(-1, 3).T
        v_0_3d = (K_1 @ v_0_f).T
        self.vertices = v_0_3d * z[..., None]
        self.surf = pv.PolyData(self.vertices, self.facets)





