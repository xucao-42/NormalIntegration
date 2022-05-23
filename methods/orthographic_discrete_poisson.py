import sys
sys.path.append("..")
sys.path.append(".")

from scipy.sparse.linalg import lsqr, cg
from utils import *
import pyvista as pv
import time
import pyamg

def generate_dx_dy(mask, step_size=1):
    # pixel coordinates
    # ^ vertical positive
    # |
    # |
    # |
    # o ---> horizontal positive
    pixel_idx = np.zeros_like(mask, dtype=int)
    pixel_idx[mask] = np.arange(np.sum(mask))
    num_pixel = np.sum(mask)

    has_left_mask = np.logical_and(move_right(mask), mask)
    has_right_mask = np.logical_and(move_left(mask), mask)
    has_bottom_mask = np.logical_and(move_top(mask), mask)
    has_top_mask = np.logical_and(move_bottom(mask), mask)

    data_term = [-1] * np.sum(has_left_mask) + [1] * np.sum(has_left_mask)
    row_idx = pixel_idx[has_left_mask]   # only the pixels having left neighbors have [-1, 1] in that row
    row_idx = np.tile(row_idx, 2)
    col_idx = np.concatenate((pixel_idx[move_left(has_left_mask)], pixel_idx[has_left_mask]))
    D_horizontal_neg = coo_matrix((data_term, (row_idx, col_idx)), shape=(num_pixel, num_pixel))

    data_term = [-1] * np.sum(has_right_mask) + [1] * np.sum(has_right_mask)
    row_idx = pixel_idx[has_right_mask]
    row_idx = np.tile(row_idx, 2)
    col_idx = np.concatenate((pixel_idx[has_right_mask], pixel_idx[move_right(has_right_mask)]))
    D_horizontal_pos = coo_matrix((data_term, (row_idx, col_idx)), shape=(num_pixel, num_pixel))

    data_term = [-1] * np.sum(has_top_mask) + [1] * np.sum(has_top_mask)
    row_idx = pixel_idx[has_top_mask]
    row_idx = np.tile(row_idx, 2)
    col_idx = np.concatenate((pixel_idx[has_top_mask], pixel_idx[move_top(has_top_mask)]))
    D_vertical_pos = coo_matrix((data_term, (row_idx, col_idx)), shape=(num_pixel, num_pixel))

    data_term = [-1] * np.sum(has_bottom_mask) + [1] * np.sum(has_bottom_mask)
    row_idx = pixel_idx[has_bottom_mask]
    row_idx = np.tile(row_idx, 2)
    col_idx = np.concatenate((pixel_idx[move_bottom(has_bottom_mask)], pixel_idx[has_bottom_mask]))
    D_vertical_neg = coo_matrix((data_term, (row_idx, col_idx)), shape=(num_pixel, num_pixel))

    return D_horizontal_pos / step_size, D_horizontal_neg / step_size, D_vertical_pos / step_size, D_vertical_neg / step_size


class OrthographicPoisson:
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
        self.method_name = "orthographic_poisson"
        print("running {}...".format(self.method_name))
        method_start = time.time()

        # Eq. (4) in "Normal Integration: A Survey."
        p = - data.n[data.mask, 0] / data.n[data.mask, 2]
        q = - data.n[data.mask, 1] / data.n[data.mask, 2]

        # Eqs. (23) and (24) in "Variational Methods for Normal Integration."
        # w/o depth prior
        dvp, dvn, dup, dun = generate_dx_dy(data.mask, data.step_size)
        A = 0.5 * (dup.T @ dup + dun.T @ dun + dvp.T @ dvp + dvn.T @ dvn)
        b = 0.5 * (dup.T + dun.T) @ p + 0.5 * (dvp.T + dvn.T) @ q

        # There should be faster solvers.
        solver_start = time.time()
        # z = lsqr(A, b, atol=1e-17, btol=1e-17)[0]
        z, _ = cg(A, b, maxiter=1000, tol=1e-9)
        # z = pyamg.solve(A, b, tol=1e-17, verb=False)
        solver_end = time.time()

        self.solver_runtime = solver_end - solver_start
        self.residual = A @ z - b

        method_end = time.time()
        self.total_runtime = method_end - method_start

        self.depth_map = np.ones_like(data.mask, dtype=np.float) * np.nan
        self.depth_map[data.mask] = z

        # create a mesh model from the depth map
        self.facets = construct_facets_from_depth_map_mask(data.mask)
        self.vertices = construct_vertices_from_depth_map_and_mask(data.mask, self.depth_map, data.step_size)
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
    result = OrthographicPoisson(data)

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


