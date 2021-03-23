# discritization follows
# Variational Methods for Normal Integration

# perspective method follows
# Normal Integration: A Survey
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr, spsolve
from utils import construct_facet_for_depth, hide_all_plot
import pyvista as pv
import os

def generate_dx_dy(mask):

    # ^ vertical positive
    # |
    # |
    # |
    # o ---> horizontal positive
    all_depth_idx = np.zeros_like(mask, dtype=np.int)
    all_depth_idx[mask] = np.arange(np.sum(mask))
    num_depth = np.sum(mask)

    move_left_mask = np.pad(mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]
    move_right_mask = np.pad(mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]
    move_top_mask = np.pad(mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]
    move_bottom_mask = np.pad(mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]

    num_neighbour_map = np.sum(np.concatenate([move_left_mask[..., None],
                           move_right_mask[..., None],
                           move_top_mask[..., None],
                           move_bottom_mask[..., None]], -1), axis=-1)
    num_neighbour_map[~mask] = 0

    has_left_mask = np.logical_and(move_right_mask, mask)
    has_right_mask = np.logical_and(move_left_mask, mask)
    has_bottom_mask = np.logical_and(move_top_mask, mask)
    has_top_mask = np.logical_and(move_bottom_mask, mask)

    has_left_mask_left = np.pad(has_left_mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]
    has_right_mask_right = np.pad(has_right_mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]
    has_bottom_mask_bottom = np.pad(has_bottom_mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]
    has_top_mask_top = np.pad(has_top_mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]

    num_has_left = np.sum(has_left_mask)
    num_has_right = np.sum(has_right_mask)
    num_has_top = np.sum(has_top_mask)
    num_has_bottom = np.sum(has_bottom_mask)

    data_term = [-1] * num_has_left + [1] * num_has_left
    row_idx = all_depth_idx[has_left_mask]
    row_idx = np.tile(row_idx, 2)
    col_idx = np.concatenate((all_depth_idx[has_left_mask_left], all_depth_idx[has_left_mask]))
    d_horizontal_neg = coo_matrix((data_term, (row_idx, col_idx)), shape=(num_depth, num_depth))

    data_term = [-1] * num_has_right + [1] * num_has_right
    row_idx = all_depth_idx[has_right_mask]
    row_idx = np.tile(row_idx, 2)
    col_idx = np.concatenate((all_depth_idx[has_right_mask], all_depth_idx[has_right_mask_right]))
    d_horizontal_pos = coo_matrix((data_term, (row_idx, col_idx)), shape=(num_depth, num_depth))
    
    data_term = [-1] * num_has_top + [1] * num_has_top
    row_idx = all_depth_idx[has_top_mask]
    row_idx = np.tile(row_idx, 2)
    col_idx = np.concatenate((all_depth_idx[has_top_mask], all_depth_idx[has_top_mask_top]))
    d_vertical_pos = coo_matrix((data_term, (row_idx, col_idx)), shape=(num_depth, num_depth))

    data_term = [-1] * num_has_bottom + [1] * num_has_bottom
    row_idx = all_depth_idx[has_bottom_mask]
    row_idx = np.tile(row_idx, 2)
    col_idx = np.concatenate((all_depth_idx[has_bottom_mask_bottom], all_depth_idx[has_bottom_mask]))
    d_vertical_neg = coo_matrix((data_term, (row_idx, col_idx)), shape=(num_depth, num_depth))

    return d_horizontal_pos, d_horizontal_neg, d_vertical_pos, d_vertical_neg


class DiscretePoissonPerspective:
    # working on camera coordinate
    # x
    # |  z
    # | /
    # |/
    # o ---y
    def __init__(self, data, setting):
        self.method = "perspective_discrete_poisson"
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

        dhp, dhn, dvp, dvn = generate_dx_dy(mask)
        Dh = 0.5 * (dhp.T + dhn.T)
        Dv = 0.5 * (dvp.T + dvn.T)
        A = 0.5 * (dhp.T @ dhp + dhn.T @ dhn + dvp.T @ dvp + dvn.T @ dvn)
        b = Dh @ q_tilde[mask] + Dv @ p_tilde[mask]

        z_tilde = lsqr(A, b)[0]

        z = np.exp(z_tilde)
        z_map = np.ones_like(mask, dtype=np.float) * np.nan
        z_map[mask] = z
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



