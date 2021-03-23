import numpy as np
from scipy.sparse.linalg import lsqr
from utils import construct_facet_for_depth, hide_all_plot
import pyvista as pv
import os
from scipy.sparse import coo_matrix, vstack, diags, identity
from scipy.sparse.linalg import eigsh

from scipy.spatial import KDTree


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

    return D_vertical, D_horizontal


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



class ZhuCDPerspective:
    # working on camera coordinate
    # x
    # |  z
    # | /
    # |/
    # o ---y
    def __init__(self, data, setting):
        self.method = "perspective_zhu_and_smith_cd"
        mask = data.mask
        num_pixel = np.sum(mask)  # number of pixels in region of interest
        H, W = mask.shape
        yy, xx = np.meshgrid(range(W), range(H))
        xx = np.max(xx) - xx
        xx = xx.astype(np.float)
        yy = yy.astype(np.float)

        ox = data.K[0, 2]
        oy = data.K[1, 2]
        fx = data.K[0, 0]
        fy = data.K[1, 1]

        u_prime = (xx - ox)[mask]
        v_prime = (yy - oy)[mask]

        # search for nearest neighbourhood pixels for each pixel in image coordinate
        try:
            self.neighbors_idx = data.neighbour_idx
        except:
            img_coordinate = np.concatenate([u_prime[..., None],
                                             v_prime[..., None]], axis=-1)
            query_tree = KDTree(img_coordinate, leafsize=1)
            _, neighbors_idx = query_tree.query(img_coordinate, k=setting.num_neighbor)
            self.neighbors_idx = np.sort(neighbors_idx, axis=-1)

        # retrieve neighbourhood points image coordinate
        center_u = u_prime
        center_v = v_prime
        neighbour_u = u_prime[neighbors_idx]
        neighbour_v = v_prime[neighbors_idx]
        poly_list = []
        order_list = []
        for i in range(setting.polynomial_order + 1):
            for j in range(setting.polynomial_order + 1 - i):
                order_list.append((i, j))
                c_i = ((neighbour_u - center_u[..., None]) ** i) * ((neighbour_v - center_v[..., None]) ** j)
                poly_list.append(c_i[..., None])
        C = np.concatenate(poly_list, axis=-1)
        C_pinv = np.linalg.pinv(C)  # num_pixels x num_polynomials x num_neighbour
        a00 = C_pinv[:, order_list.index((0, 0)), :].flatten()  # num_pixels x num_neighbour. smoothness term

        # construct smoothing kernel
        row_id = np.arange(num_pixel)
        row_id = np.repeat(row_id, setting.num_neighbor)
        col_id = neighbors_idx.flatten()

        S = coo_matrix((a00, (row_id, col_id)), shape=(num_pixel, num_pixel))

        U = diags(u_prime)
        V = diags(v_prime)

        if setting.add_noise and setting.add_outlier:
            n1 = data.n_outlier_noise[..., 0][mask]
            n2 = data.n_outlier_noise[..., 1][mask]
            n3 = data.n_outlier_noise[..., 2][mask]
        elif setting.add_noise:
            n1 = data.n_noise[..., 0][mask]
            n2 = data.n_noise[..., 1][mask]
            n3 = data.n_noise[..., 2][mask]
        elif setting.add_outlier:
            n1 = data.n_outlier[..., 0][mask]
            n2 = data.n_outlier[..., 1][mask]
            n3 = data.n_outlier[..., 2][mask]
        else:
            n1 = data.n[..., 0][mask]
            n2 = data.n[..., 1][mask]
            n3 = data.n[..., 2][mask]

        N = vstack([diags(n1),
                    diags(n2),
                    diags(n3)]).T

        D_u, D_v = generate_dx_dy_wb(mask, 1)

        Tx = vstack([(U @ D_u + identity(num_pixel)) / fx,
                     V @ D_u / fy,
                     D_u])
        Ty = vstack([U @ D_v / fx,
                     (V @ D_v + identity(num_pixel)) / fy,
                     D_v])

        A = vstack([N @ Tx,
                    N @ Ty,
                    setting.lambda_smooth * (S - identity(num_pixel))])

        # forward_res = A @ data.depth_gt[mask]
        _, z = eigsh(A.T @ A, k=1, sigma=0, which="LM")

        # if np.sum(z) < 0:
        #     z = -z
        z_map = np.zeros_like(mask, dtype=np.float)
        z_map[mask] = np.squeeze(z)
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
        self.vertices = v_0_3d * z
        self.surf = pv.PolyData(self.vertices, self.facets)
