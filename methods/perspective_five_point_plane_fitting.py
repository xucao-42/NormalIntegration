import sys
sys.path.append("..")
sys.path.append(".")

import numpy as np
from utils import *
from scipy.sparse import coo_matrix, hstack
from scipy.sparse.linalg import eigsh
import pyvista as pv
import time


class PerspectiveFivePointPlaneFitting:
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
        self.method_name = "perspective_five_point_plane_fitting"
        print("running {}...".format(self.method_name))

        method_start = time.time()

        H, W = data.mask.shape
        vv, uu = np.meshgrid(range(W), range(H))
        uu = np.flip(uu, axis=0)

        # pixel_coordinates = np.stack((uu[data.mask], vv[data.mask]), axis=-1)
        # from sklearn.neighbors import KDTree
        # tree = KDTree(pixel_coordinates)
        # neighbor_pixel_ids = tree.query_radius(pixel_coordinates, r=1 + 1e-7)

        # For each pixel, search for its neighbor pixel indices and store them as an item in the list neighbor_pixel_ids
        # including itself's index
        pixel_idx = np.zeros_like(data.mask, dtype=int)
        pixel_idx[data.mask] = np.arange(np.sum(data.mask)) + 1  # pixel idx starts from 1

        expand_mask = np.pad(data.mask, 1, "constant", constant_values=0)
        expand_pixel_idx = np.pad(pixel_idx, 1, "constant", constant_values=0)

        top_neighbor = expand_pixel_idx[move_top(expand_mask)]
        bottom_neighbor = expand_pixel_idx[move_bottom(expand_mask)]
        left_neighbor = expand_pixel_idx[move_left(expand_mask)]
        right_neighbor = expand_pixel_idx[move_right(expand_mask)]

        neighbor_pixel_ids = np.stack((pixel_idx[data.mask], top_neighbor, bottom_neighbor, left_neighbor, right_neighbor), axis=-1)
        neighbor_pixel_ids = [i[i!=0] - 1 for i in neighbor_pixel_ids]  # pixel idx starts from 0 now

        # construct the system matrix A based on the list of neighbor pixel indices
        num_neighbor_list = [len(i) for i in neighbor_pixel_ids]
        num_plane_equations = sum(num_neighbor_list)
        num_normals = np.sum(data.mask)

        # Eq. (14) in "Normal Integration via Inverse Plane Fitting with Minimum Point-to-Plane Distance"
        u_tilde = np.zeros((H, W, 3))
        u_tilde[..., 0] = uu
        u_tilde[..., 1] = vv
        u_tilde[..., 2] = 1
        u_tilde = u_tilde[data.mask].T
        K_inv = np.linalg.inv(data.K)
        p_tilde = (K_inv @ u_tilde).T

        col_idx = np.concatenate(neighbor_pixel_ids)
        row_idx = np.arange(num_plane_equations)

        n_vec = data.n[data.mask]  # p x 3
        n_vec = np.repeat(n_vec, num_neighbor_list, axis=0)
        p_vec = p_tilde[col_idx]
        A_left_data = np.sum(n_vec * p_vec, axis=-1)
        A_left = coo_matrix((A_left_data, (row_idx, col_idx)))

        all_ones = np.ones(num_plane_equations)
        A_right_col = np.arange(num_normals)
        A_right_col = np.repeat(A_right_col, num_neighbor_list)
        A_right = coo_matrix((all_ones, (row_idx, A_right_col)))

        A = hstack([A_left, A_right])

        solver_start = time.time()
        _, x = eigsh(A.T @ A, k=1, sigma=0, which="LM")
        solver_end = time.time()

        self.solver_runtime = solver_end - solver_start
        method_end = time.time()
        self.total_runtime = method_end - method_start

        self.depth_map = np.ones_like(data.mask, dtype=np.float) * np.nan
        self.depth_map[data.mask] = np.squeeze(x[:num_normals])

        # construct a mesh from the depth map
        self.vertices = p_tilde * x[:num_normals]
        self.facets = construct_facets_from_depth_map_mask(data.mask)
        self.surface = pv.PolyData(self.vertices, self.facets)


if __name__ == "__main__":
    import argparse
    from data.data_loader import data_loader
    import cv2
    import os
    from utils import crop_a_set_of_images, file_path

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=file_path)
    parser.add_argument('-s', '--save_normal', type=bool, default=True)
    par = parser.parse_args()

    data = data_loader(par.path)
    result = PerspectiveFivePointPlaneFitting(data)

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


