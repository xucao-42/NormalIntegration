import sys
sys.path.append("..")
sys.path.append(".")

from utils import *
from scipy.sparse import coo_matrix, hstack
import pyvista as pv
from scipy.sparse.linalg import lsqr, cg
import time

class OrthographicFourPoint:
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
        self.method_name = "orthographic_four_point_plane_fitting"
        print("running {}...".format(self.method_name))
        method_start = time.time()
        facet_H, facet_W = data.mask.shape

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

        vertex_idx = np.zeros((facet_H + 1, facet_W + 1), dtype=np.int)
        vertex_idx[vertex_mask] = np.arange(np.sum(vertex_mask)) + 1  # vertex indices start from 1

        num_facet = np.sum(data.mask)
        num_vertex = np.sum(vertex_mask)

        top_left_vertex = vertex_idx[top_left_mask].flatten()
        top_right_vertex = vertex_idx[top_right_mask].flatten()
        bottom_left_vertex = vertex_idx[bottom_left_mask].flatten()
        bottom_right_vertex = vertex_idx[bottom_right_mask].flatten()
        neighbor_pixel_ids = np.stack((top_left_vertex, bottom_left_vertex, bottom_right_vertex, top_right_vertex), axis=-1)

        vv, uu = np.meshgrid(range(facet_W + 1), range(facet_H + 1))
        uu = np.flip(uu, axis=0)
        uu = (uu - 0.5) * data.step_size
        vv = (vv - 0.5) * data.step_size

        v_0 = np.zeros((facet_H + 1, facet_W + 1, 3))
        v_0[..., 0] = uu
        v_0[..., 1] = vv

        # construct the left and the right part of A
        num_plane_equations = np.sum(neighbor_pixel_ids != 0)
        row_idx = np.arange(num_plane_equations)

        nz = data.n[data.mask, 2]
        nz = np.repeat(nz, 4)
        col_idx = (neighbor_pixel_ids - 1).flatten()
        A_left = coo_matrix((nz, (row_idx, col_idx)))

        A_right_data = np.ones(num_plane_equations)
        A_right_col = np.arange(num_facet)
        A_right_col = np.repeat(A_right_col, 4)
        A_right = coo_matrix((A_right_data, (row_idx, A_right_col)))

        A = hstack([A_left, A_right])

        u_vec = uu[vertex_mask]
        v_vec = vv[vertex_mask]
        nx = data.n[data.mask, 0]
        ny = data.n[data.mask, 1]
        nx = np.repeat(nx, 4)
        ny = np.repeat(ny, 4)

        u = u_vec[neighbor_pixel_ids - 1].flatten()
        v = v_vec[neighbor_pixel_ids - 1].flatten()
        b = - u * nx - v * ny

        solver_start = time.time()
        # z = lsqr(A, b, atol=1e-17, btol=1e-17)[0]
        z, _ = cg(A.T @ A, A.T @ b, maxiter=1000, tol=1e-9)

        solver_end = time.time()

        self.solver_runtime = solver_end - solver_start
        method_end = time.time()
        self.total_runtime = method_end - method_start

        self.residual = A @ z - b

        plane_displacements = np.squeeze(z[num_vertex:])

        # center directions are used for extrcting depth values at Omega_n
        center_vv, center_uu = np.meshgrid(range(facet_W), range(facet_H))
        center_uu = np.flip(center_uu, axis=0)
        center_uu = center_uu[data.mask] * data.step_size
        center_vv = center_vv[data.mask] * data.step_size

        center_depth = (- plane_displacements - center_uu * data.n[data.mask, 0] -
                        center_vv * data.n[data.mask, 1]) / data.n[data.mask, 2]

        self.depth_map = np.ones_like(data.mask, dtype=np.float) * np.nan
        self.depth_map[data.mask] = center_depth
        method_end = time.time()
        self.total_runtime = method_end - method_start

        # create a mesh model from the depth map
        self.vertices = construct_vertices_from_depth_map_and_mask(data.mask, self.depth_map, data.step_size)
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
    result = OrthographicFourPoint(data)

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
