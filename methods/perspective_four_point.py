import numpy as np
from utils import construct_facet_for_depth, hide_all_plot
from scipy.sparse import coo_matrix, hstack, diags, identity
from scipy.sparse.linalg import eigsh
import pyvista as pv


class FourPointPlaneFittingPerspective:
    # working on camera coordinate
    # x
    # |  z
    # | /
    # |/
    # o ---y
    def __init__(self, data, setting):
        self.method = "perspective_four_point_plane_fitting"

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

        # facet_id_vertice_id is for constructing a mesh
        top_left_vertex = vertex_idx[top_left_mask].flatten()
        top_right_vertex = vertex_idx[top_right_mask].flatten()
        bottom_left_vertex = vertex_idx[bottom_left_mask].flatten()
        bottom_right_vertex = vertex_idx[bottom_right_mask].flatten()
        facet_id_vertice_id = np.hstack((top_left_vertex[:, None],
                                         bottom_left_vertex[:, None],
                                         bottom_right_vertex[:, None],
                                         top_right_vertex[:, None]))  # start from 1

        K_1 = np.linalg.inv(data.K)

        yy, xx = np.meshgrid(range(facet_W + 1), range(facet_H + 1))
        xx = np.max(xx) - xx
        xx = xx.astype(np.float)
        yy = yy.astype(np.float)
        xx -= 0.5
        yy -= 0.5

        v_0 = np.zeros((facet_H + 1, facet_W + 1, 3))
        v_0[..., 0] = xx
        v_0[..., 1] = yy
        v_0[..., 2] = 1
        v_0_f = v_0.reshape(-1, 3).T
        v_0_3d = (K_1 @ v_0_f).T.reshape((facet_H + 1, facet_W + 1, 3))
        vertex_directions = v_0_3d[vertex_mask]

        # center directions are used for extrcting depth values at Omega_n
        center_yy, center_xx = np.meshgrid(range(facet_W), range(facet_H))
        center_xx = np.max(center_xx) - center_xx
        center_xx = center_xx.astype(np.float)
        center_yy = center_yy.astype(np.float)

        center_v0 = np.zeros((facet_H, facet_W, 3))
        center_v0[..., 0] = center_xx
        center_v0[..., 1] = center_yy
        center_v0[..., 2] = 1
        center_v0f = center_v0.reshape(-1, 3).T
        center_directions = (K_1 @ center_v0f).T.reshape((facet_H, facet_W, 3))
        center_directions = center_directions[facet_mask]

        # construct the left and the right part of A
        if setting.add_noise and setting.add_outlier:
            n_vec = data.n_outlier_noise[facet_mask]
        elif setting.add_noise:
            n_vec = data.n_noise[facet_mask]
        elif setting.add_outlier:
            n_vec = data.n_outlier[facet_mask]
        else:
            n_vec = data.n[facet_mask]

        v_vec = vertex_directions[facet_id_vertice_id -1]
        data_ = np.sum(n_vec[:, None, :] * v_vec, axis=-1).flatten()

        row_idx = np.arange(num_facet * 4)

        A_right_data = np.ones(num_facet * 4)
        A_right_col = np.arange(num_facet)
        A_right_col = np.repeat(A_right_col, 4)
        A_right = coo_matrix((A_right_data, (row_idx, A_right_col)))

        col_idx = (facet_id_vertice_id - 1).flatten()
        A_left = coo_matrix((data_, (row_idx, col_idx)))
        A = hstack([A_left, A_right])

        # svd on A
        _, x = eigsh(A.T @ A, k=1, sigma=0, which="LM")

        vertex_depth = np.squeeze(x[:num_vertex])
        plane_displacement = np.squeeze(x[num_vertex:])

        if setting.add_noise and setting.add_outlier:
            center_depth = - plane_displacement / (np.sum(data.n_outlier_noise[facet_mask] * center_directions, axis=-1))
        elif setting.add_noise:
            center_depth = - plane_displacement / (np.sum(data.n_noise[facet_mask] * center_directions, axis=-1))
        elif setting.add_outlier:
            center_depth = - plane_displacement / (np.sum(data.n_outlier[facet_mask] * center_directions, axis=-1))
        else:
            center_depth = - plane_displacement / (np.sum(data.n[facet_mask] * center_directions, axis=-1))

        center_points = center_depth[:, None] * center_directions

        self.vertices = vertex_directions * vertex_depth[..., None]
        self.vertices -= np.mean(self.vertices, 0, keepdims=1)
        self.facets = np.hstack((np.ones((num_facet, 1)) * 4, facet_id_vertice_id - 1)).astype(np.int)

        self.depth_facets = construct_facet_for_depth(facet_mask)
        self.surf = pv.PolyData(center_points, self.depth_facets)

        depth_map = np.ones_like(facet_mask, dtype=np.float) * np.nan
        depth_map[facet_mask] = center_depth
        self.depth = depth_map


