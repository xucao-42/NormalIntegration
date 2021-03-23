import numpy as np
from utils import construct_facet_for_depth, hide_all_plot
from scipy.sparse import coo_matrix, hstack, diags, identity, vstack
import pyvista as pv
from scipy.sparse.linalg import lsqr


class DiscreteGeometryProcessingOrthographic:
    # working on camera coordinate
    # x
    # |  z
    # | /
    # |/
    # o ---y
    def __init__(self, data, setting):
        self.method_name = "orthographic_discrete_geometry_processing"

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
        except :
            n_vec = data.n[facet_mask]

        nx = n_vec[..., 0]
        ny = n_vec[..., 1]
        nz = n_vec[..., 2]

        projection_top_left = - (0.5 * nx - 0.5 * ny) / nz
        projection_bottom_left = - (- 0.5 * nx - 0.5 * ny) / nz
        projection_bottom_right = - (- 0.5 * nx + 0.5 * ny) / nz
        projection_top_right = - (0.5 * nx + 0.5 * ny) / nz

        row_idx = np.arange(num_facet)
        row_idx = np.repeat(row_idx, 4)
        col_idx = facet_id_vertice_id.flatten() - 1

        data_term = [0.75, -0.25, -0.25, -0.25] * num_facet
        A_top_left = coo_matrix((data_term, (row_idx, col_idx)), shape=(num_facet, num_vertex))

        data_term = [-0.25, 0.75, -0.25, -0.25] * num_facet
        A_bottom_left = coo_matrix((data_term, (row_idx, col_idx)), shape=(num_facet, num_vertex))

        data_term = [-0.25, -0.25, 0.75, -0.25] * num_facet
        A_bottom_right = coo_matrix((data_term, (row_idx, col_idx)), shape=(num_facet, num_vertex))

        data_term = [-0.25, -0.25, -0.25, 0.75] * num_facet
        A_top_right = coo_matrix((data_term, (row_idx, col_idx)), shape=(num_facet, num_vertex))

        A = vstack([A_top_left, A_bottom_left, A_bottom_right, A_top_right])

        b = np.concatenate((projection_top_left,
                            projection_bottom_left,
                            projection_bottom_right,
                            projection_top_right)) * data.step_size

        z = lsqr(A, b, atol=1e-17, btol=1e-17)[0]
        self.res = A @ z - b

        # vertex_depth = np.squeeze(z[:num_vertex])
        depth_facet = z[facet_id_vertice_id - 1]
        center_depth = np.mean(depth_facet, axis=-1)
        center_points[facet_mask, 2] = center_depth

        self.depth_facets = construct_facet_for_depth(facet_mask)
        self.surf = pv.PolyData(center_points[facet_mask], self.depth_facets)

        depth_map = np.ones_like(facet_mask, dtype=np.float) * np.nan
        depth_map[facet_mask] = center_depth
        self.depth = depth_map


# if __name__ == "__main__":
#     from data_sphere import sphere_o as obj
#     import matplotlib.pyplot as plt
#     import time, os
#
#     class Setting:
#         pass
#
#     setting = Setting()
#     st_time = str(time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time())))
#     setting.save_dir = "../selected_results/" + st_time
#     if not os.path.exists(setting.save_dir):
#         os.mkdir(setting.save_dir)
#     setting.add_noise = False
#     setting.add_outlier = False
#
#     z_est = XieOrthographic(obj, setting)
#
#     offset = np.nanmean(obj.depth_gt - z_est.depth)
#     offset_depth = z_est.depth + offset
#     rmse_map = (offset_depth - obj.depth_gt) ** 2
#     rmse = np.sqrt(np.nanmean(rmse_map))
#
#     hide_all_plot(rmse_map, vmax=None, colorbar=True,
#                   fname=os.path.join(setting.save_dir, "rmse_{:.5f}.png".format(rmse)))
