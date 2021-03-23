# discritization follows
# Variational Methods for Normal Integration

# perspective method follows
# Normal Integration: A Survey
from scipy.sparse import vstack
from scipy.sparse.linalg import lsqr
from utils import *
import pyvista as pv
import os


def generate_dx_dy(mask, step_size):
    # ^ vertical positive
    # |
    # |
    # |
    # o ---> horizontal positive
    all_depth_idx = np.zeros_like(mask, dtype=np.int)
    all_depth_idx[mask] = np.arange(np.sum(mask))
    num_depth = np.sum(mask)

    num_neighbour_map = np.sum(np.concatenate([move_left(mask)[..., None],
                                               move_right(mask)[..., None],
                                               move_top(mask)[..., None],
                                               move_bottom(mask)[..., None]], -1), axis=-1)
    num_neighbour_map[~mask] = 0

    has_left_mask = np.logical_and(move_right(mask), mask)
    has_right_mask = np.logical_and(move_left(mask), mask)
    has_bottom_mask = np.logical_and(move_top(mask), mask)
    has_top_mask = np.logical_and(move_bottom(mask), mask)

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

    return d_horizontal_pos / step_size, d_horizontal_neg / step_size, d_vertical_pos / step_size, d_vertical_neg / step_size


class PoissonOrthographic:
    # x
    # |  z
    # | /
    # |/
    # o ---y
    def __init__(self, data, setting):
        self.method_name = "orthographic_poisson"
        mask = data.mask

        p = - data.n_used[mask, 0] / data.n_used[mask, 2]
        q = - data.n_used[mask, 1] / data.n_used[mask, 2]

        dhp, dhn, dvp, dvn = generate_dx_dy(mask, data.step_size)
        Dh = 0.5 * (dhp.T + dhn.T)
        Dv = 0.5 * (dvp.T + dvn.T)
        A = 0.5 * (dhp.T @ dhp + dhn.T @ dhn + dvp.T @ dvp + dvn.T @ dvn)
        b = Dh @ q + Dv @ p

        z = lsqr(A, b, atol=1e-17, btol=1e-17)[0]
        self.res = A @ z - b

        z_map = np.ones_like(mask, dtype=np.float) * np.nan
        z_map[mask] = z
        self.depth = z_map

        # construct a mesh from the depth map
        self.facets = construct_facet_for_depth(mask)
        H, W = mask.shape
        yy, xx = np.meshgrid(range(W), range(H))
        xx = np.max(xx) - xx
        xx = xx.astype(np.float)
        yy = yy.astype(np.float)
        xx *= data.step_size
        yy *= data.step_size
        v_0 = np.zeros((H, W, 3))
        v_0[..., 0] = xx
        v_0[..., 1] = yy
        v_0[..., 2] = z_map
        self.vertices = v_0[mask].reshape(-1, 3)
        self.surf = pv.PolyData(self.vertices, self.facets)




