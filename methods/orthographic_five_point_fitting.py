from utils import *
from scipy.sparse import coo_matrix, hstack
import pyvista as pv
from scipy.sparse.linalg import lsqr

class FivePointOthographic:
    # camera coordinate
    # x
    # |  z
    # | /
    # |/
    # o ---y
    def __init__(self, data, setting):
        self.method_name = "orthographic_five_point_plane_fitting"

        normal_mask = data.mask
        H, W = normal_mask.shape

        pixel_idx = np.zeros_like(normal_mask, dtype=np.int)
        pixel_idx[normal_mask] = np.arange(np.sum(normal_mask)) + 1  # pixel idx begin from 1

        expand_mask = np.pad(normal_mask, 1, "constant", constant_values=0)
        expand_pixel_idx = np.pad(pixel_idx, 1, "constant", constant_values=0)

        num_normal = np.sum(normal_mask)
        top_neighbor = expand_pixel_idx[move_top(expand_mask)].flatten()
        bottom_neighbor = expand_pixel_idx[move_bottom(expand_mask)].flatten()
        left_neighbor = expand_pixel_idx[move_left(expand_mask)].flatten()
        right_neighbor = expand_pixel_idx[move_right(expand_mask)].flatten()

        neighbor_id = np.hstack((pixel_idx[normal_mask][:, None],
                                 top_neighbor[:, None],
                                 bottom_neighbor[:, None],
                                 left_neighbor[:, None],
                                 right_neighbor[:, None]))  # start from 1

        yy, xx = np.meshgrid(range(W), range(H))
        xx = np.max(xx) - xx
        xx = xx.astype(np.float)
        yy = yy.astype(np.float)
        xx *= data.step_size
        yy *= data.step_size

        # construct the left and the right part of A
        try:
            n_vec = data.n_used[normal_mask]
        except:
            n_vec = data.n[normal_mask]

        num_eq = np.sum(neighbor_id != 0)
        row_idx = np.arange(num_eq)

        repeat = np.sum(neighbor_id != 0, axis=-1)
        data_ = n_vec[:, 2]
        data_ = np.repeat(data_, repeat)
        col_idx = (neighbor_id - 1).flatten()
        col_idx = col_idx[col_idx != -1]
        A_left = coo_matrix((data_, (row_idx, col_idx)))

        A_right_data = np.ones(num_eq)
        A_right_col = np.arange(num_normal)
        A_right_col = np.repeat(A_right_col, repeat)
        A_right = coo_matrix((A_right_data, (row_idx, A_right_col)))

        A = hstack([A_left, A_right])

        u_vec = xx[normal_mask]
        v_vec = yy[normal_mask]
        n1 = n_vec[:, 0]
        n2 = n_vec[:, 1]
        u_vec = np.insert(u_vec, 0, 0)
        v_vec = np.insert(v_vec, 0, 0)
        n1 = np.repeat(n1, 5)
        n2 = np.repeat(n2, 5)

        u_vec = u_vec[neighbor_id].flatten()
        v_vec = v_vec[neighbor_id].flatten()
        b = (- u_vec * n1 - v_vec * n2)[neighbor_id.flatten() != 0]

        z = lsqr(A, b, atol=1e-17, btol=1e-17)[0]
        self.res = A @ z - b

        z_map = np.ones_like(normal_mask, dtype=np.float) * np.nan
        z_map[normal_mask] = z[:num_normal]
        self.depth = z_map

        # construct a mesh from the depth map
        self.facets = construct_facet_for_depth(normal_mask)
        v_0 = np.zeros((H, W, 3))
        v_0[..., 0] = xx
        v_0[..., 1] = yy
        v_0[..., 2] = z_map
        self.vertices = v_0[normal_mask].reshape(-1, 3)
        self.surf = pv.PolyData(self.vertices, self.facets)


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
#     setting.include_center_point = True
#
#     z_est = FivePointOthographic(obj, setting)
#
#     offset = np.nanmean(obj.depth_gt - z_est.depth)
#     offset_depth = z_est.depth + offset
#     rmse_map = (offset_depth - obj.depth_gt) ** 2
#     rmse = np.sqrt(np.nanmean(rmse_map))
#
#     hide_all_plot(rmse_map, vmax=None, colorbar=True,
#                   fname=os.path.join(setting.save_dir, "rmse_{:.5f}.png".format(rmse)))
