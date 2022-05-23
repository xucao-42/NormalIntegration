import sys
sys.path.append("..")
sys.path.append(".")

import numpy as np
from utils import construct_facets_from_depth_map_mask
from scipy.sparse import coo_matrix, hstack, diags, identity
from scipy.sparse.linalg import eigsh
import pyvista as pv
import time


class PerspectiveFourPointPlaneFitting:
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
        self.method_name = "perspective_four_point_plane_fitting"
        print("running {}...".format(self.method_name))
        method_start = time.time()

        H, W = data.mask.shape

        facet_idx = np.zeros_like(data.mask, dtype=np.int)
        facet_idx[data.mask] = np.arange(np.sum(data.mask)) + 1  # facet idx begin from 1

        top_left_mask = np.pad(data.mask, ((0, 1), (0, 1)), "constant", constant_values=0)
        top_right_mask = np.pad(data.mask, ((0, 1), (1, 0)), "constant", constant_values=0)
        bottom_left_mask = np.pad(data.mask, ((1, 0), (0, 1)), "constant", constant_values=0)
        bottom_right_mask = np.pad(data.mask, ((1, 0), (1, 0)), "constant", constant_values=0)

        vertex_mask = np.logical_or.reduce((top_right_mask,
                                            top_left_mask,
                                            bottom_right_mask,
                                            bottom_left_mask))

        vertex_idx = np.zeros((H + 1, W + 1), dtype=np.int)
        vertex_idx[vertex_mask] = np.arange(np.sum(vertex_mask)) + 1  # vertex idx begin from 1

        num_facet = np.sum(data.mask)
        num_vertex = np.sum(vertex_mask)

        top_left_vertex = vertex_idx[top_left_mask].flatten()
        top_right_vertex = vertex_idx[top_right_mask].flatten()
        bottom_left_vertex = vertex_idx[bottom_left_mask].flatten()
        bottom_right_vertex = vertex_idx[bottom_right_mask].flatten()
        neighbor_pixel_ids = np.hstack((top_left_vertex[:, np.newaxis],
                                        bottom_left_vertex[:, np.newaxis],
                                        bottom_right_vertex[:, np.newaxis],
                                        top_right_vertex[:, np.newaxis]))  # start from 1

        K_inv = np.linalg.inv(data.K)

        vv, uu = np.meshgrid(range(W + 1), range(H + 1))
        uu = np.flip(uu, axis=0)
        uu = uu - 0.5
        vv = vv - 0.5

        u_tilde = np.zeros((H + 1, W + 1, 3))
        u_tilde[..., 0] = uu
        u_tilde[..., 1] = vv
        u_tilde[..., 2] = 1
        u_tilde = u_tilde[vertex_mask].T
        p_tilde = (K_inv @ u_tilde).T

        # center directions are used for extracting depth values at Omega_n
        center_yy, center_xx = np.meshgrid(range(W), range(H))
        center_xx = np.flip(center_xx, axis=0)

        center_v0 = np.zeros((H, W, 3))
        center_v0[..., 0] = center_xx
        center_v0[..., 1] = center_yy
        center_v0[..., 2] = 1
        center_v0f = center_v0[data.mask].T
        center_directions = (K_inv @ center_v0f).T

        # construct the left and the right part of A
        n_vec = data.n[data.mask]
        v_vec = p_tilde[neighbor_pixel_ids -1]
        data_ = np.sum(n_vec[:, None, :] * v_vec, axis=-1).flatten()

        row_idx = np.arange(num_facet * 4)

        A_right_data = np.ones(num_facet * 4)
        A_right_col = np.arange(num_facet)
        A_right_col = np.repeat(A_right_col, 4)
        A_right = coo_matrix((A_right_data, (row_idx, A_right_col)))

        col_idx = (neighbor_pixel_ids - 1).flatten()
        A_left = coo_matrix((data_, (row_idx, col_idx)))
        A = hstack([A_left, A_right])

        solver_start = time.time()

        _, x = eigsh(A.T @ A, k=1, sigma=0, which="LM")

        solver_end = time.time()
        self.solver_runtime = solver_end - solver_start

        method_end = time.time()
        self.total_runtime = method_end - method_start

        plane_displacements = np.squeeze(x[num_vertex:])

        center_depth = - plane_displacements / (np.sum(data.n[data.mask] * center_directions, axis=-1))
        center_points = center_depth[:, None] * center_directions

        self.depth_map = np.ones_like(data.mask, dtype=np.float) * np.nan
        self.depth_map[data.mask] = center_depth

        # construct a mesh from the depth map
        self.facets = construct_facets_from_depth_map_mask(data.mask)
        self.surface = pv.PolyData(center_points, self.facets)


if __name__ == "__main__":
    import argparse
    from data.data_loader import data_loader
    import cv2
    from utils import crop_a_set_of_images, file_path
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=file_path)
    parser.add_argument('-s', '--save_normal', type=bool, default=True)
    par = parser.parse_args()

    data = data_loader(par.path)
    result = PerspectiveFourPointPlaneFitting(data)

    file_dir = os.path.dirname(par.path)

    # save the estimated surface as a .ply file
    result.surface.save(os.path.join(file_dir, "est_surface_{}.ply".format(result.method_name)), binary=True)

    # save the input normal map
    if par.save_normal and os.path.isfile(par.path):
        cv2.imwrite(os.path.join(file_dir, "input_normal_map.png"), cv2.cvtColor(data.n_vis.astype(np.uint8), cv2.COLOR_BGR2RGB))

    # save the image of estimated surface
    img_path = os.path.join(file_dir, "est_surface_{}.png".format(result.method_name))
    pv.set_plot_theme("document")
    print("plotting surface ...")
    camera_pose = result.surface.plot()
    result.surface.plot(cpos=camera_pose,
                          diffuse=0.5,
                          ambient=0.5,
                          specular=0.3,
                          color="w",
                          smooth_shading=False,
                          show_scalar_bar=False,
                          show_axes=False,
                          eye_dome_lighting=True,
                          off_screen=True,
                          screenshot=img_path,
                          window_size=(1024, 768))

    crop_a_set_of_images(*[img_path])

