import sys
sys.path.append("..")
sys.path.append(".")

import numpy as np
from utils import *
from scipy.sparse import coo_matrix, vstack
import pyvista as pv
from scipy.sparse.linalg import lsqr, cg
import time

class OrthographicDiscreteGeometryProcessing:
    # an implementation of "Surface-from-Gradients: An Approach Based on Discrete Geometry Processing."
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
        self.method_name = "orthographic_discrete_geometry_processing"
        print("running {}...".format(self.method_name))
        method_start = time.time()
        facet_H, facet_W = data.mask.shape

        facet_idx = np.zeros_like(data.mask, dtype=np.int)
        facet_idx[data.mask] = np.arange(np.sum(data.mask))

        top_left_mask = np.pad(data.mask, ((0, 1), (0, 1)), "constant", constant_values=0)
        top_right_mask = np.pad(data.mask, ((0, 1), (1, 0)), "constant", constant_values=0)
        bottom_left_mask = np.pad(data.mask, ((1, 0), (0, 1)), "constant", constant_values=0)
        bottom_right_mask = np.pad(data.mask, ((1, 0), (1, 0)), "constant", constant_values=0)

        # vertex mask is (H+1, W+1)
        vertex_mask = np.logical_or.reduce((top_right_mask, top_left_mask, bottom_right_mask, bottom_left_mask))
        vertex_idx = np.zeros((facet_H + 1, facet_W + 1), dtype=np.int)
        vertex_idx[vertex_mask] = np.arange(np.sum(vertex_mask))

        num_facet = np.sum(data.mask)
        num_vertex = np.sum(vertex_mask)

        # each row in facet_id_vertice_id consists of the vertex indices belonging to the fact.
        top_left_vertex = vertex_idx[top_left_mask].flatten()
        top_right_vertex = vertex_idx[top_right_mask].flatten()
        bottom_left_vertex = vertex_idx[bottom_left_mask].flatten()
        bottom_right_vertex = vertex_idx[bottom_right_mask].flatten()
        facet_id_vertice_id = np.hstack((top_left_vertex[:, None],
                                         bottom_left_vertex[:, None],
                                         bottom_right_vertex[:, None],
                                         top_right_vertex[:, None]))

        # center directions are used for extracting depth values at \Omega_n
        center_yy, center_xx = np.meshgrid(range(facet_W), range(facet_H))
        center_xx = np.max(center_xx) - center_xx
        center_xx = center_xx[data.mask] * data.step_size
        center_yy = center_yy[data.mask] * data.step_size

        facet_center_points = np.zeros((facet_H, facet_W, 3))
        facet_center_points[data.mask, 0] = center_xx
        facet_center_points[data.mask, 1] = center_yy

        # construct the left and the right part of A
        nx = data.n[data.mask, 0]
        ny = data.n[data.mask, 1]
        nz = data.n[data.mask, 2]

        # Eq. (2) in "Surface-from-Gradients: An Approach Based on Discrete Geometry Processing."
        projection_top_left = - (0.5 * nx - 0.5 * ny) / nz
        projection_bottom_left = - (- 0.5 * nx - 0.5 * ny) / nz
        projection_bottom_right = - (- 0.5 * nx + 0.5 * ny) / nz
        projection_top_right = - (0.5 * nx + 0.5 * ny) / nz

        row_idx = np.arange(num_facet)
        row_idx = np.repeat(row_idx, 4)
        col_idx = facet_id_vertice_id.flatten()

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

        solver_start = time.time()
        # z = lsqr(A, b, atol=1e-17, btol=1e-17)[0]
        z, _ = cg(A.T @ A, A.T @ b, maxiter=1000, tol=1e-9)
        solver_end = time.time()

        self.solver_runtime = solver_end - solver_start
        method_end = time.time()
        self.total_runtime = method_end - method_start

        self.residual = A @ z - b

        # vertex_depth = np.squeeze(z[:num_vertex])
        depth_facet = z[facet_id_vertice_id]
        facet_center_depth = np.mean(depth_facet, axis=-1)
        facet_center_points[data.mask, 2] = facet_center_depth

        self.facets = construct_facets_from_depth_map_mask(data.mask)
        self.surface = pv.PolyData(facet_center_points[data.mask], self.facets)

        self.depth_map = np.ones_like(data.mask, dtype=np.float) * np.nan
        self.depth_map[data.mask] = facet_center_depth


if __name__ == "__main__":
    import argparse
    from data.data_loader import data_loader
    import cv2
    import os
    from utils import crop_a_set_of_images

    def file_path(string):
        if os.path.isfile(string):
            return string
        else:
            raise FileNotFoundError(string)

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=file_path)
    parser.add_argument('-s', '--save_normal', type=bool, default=True)
    par = parser.parse_args()

    data = data_loader(par.path)
    result = OrthographicDiscreteGeometryProcessing(data)

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