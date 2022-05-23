# discritization follows
# Variational Methods for Normal Integration

# perspective method follows
# Normal Integration: A Survey
import sys
sys.path.append("..")
sys.path.append(".")

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr, spsolve, cg
from utils import construct_facets_from_depth_map_mask, map_depth_map_to_point_clouds
import pyvista as pv
import os
from orthographic_discrete_poisson import generate_dx_dy
import time
import pyamg


class PerspectiveDiscretePoisson:
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
        self.method_name = "perspective_discrete_poisson"
        print("running {}...".format(self.method_name))
        method_start = time.time()
        H, W = data.mask.shape
        yy, xx = np.meshgrid(range(W), range(H))
        xx = np.flip(xx, axis=0)

        ox = data.K[0, 2]
        oy = data.K[1, 2]
        fx = data.K[0, 0]
        fy = data.K[1, 1]

        uu = xx - ox
        vv = yy - oy

        n1 = data.n[..., 0]
        n2 = data.n[..., 1]
        n3 = data.n[..., 2]

        # Eq. (17) in "Normal Integration: A Survey."
        p_tilde = - n1 / (uu * n1 + vv * n2 + fx * n3)
        q_tilde = - n2 / (uu * n1 + vv * n2 + fy * n3)

        # Eqs. (23) and (24) in "Variational Methods for Normal Integration."
        # w/o depth prior
        dvp, dvn, dup, dun = generate_dx_dy(data.mask)
        A = 0.5 * (dup.T @ dup + dun.T @ dun + dvp.T @ dvp + dvn.T @ dvn)
        b = 0.5 * (dup.T + dun.T) @ p_tilde[data.mask] + 0.5 * (dvp.T + dvn.T) @ q_tilde[data.mask]

        solver_start = time.time()

        # z_tilde = lsqr(A, b)[0]
        # z_tilde = pyamg.solve(A, b, tol=1e-17, verb=False)
        z_tilde, _ = cg(A, b, maxiter=1000, tol=1e-9)

        z = np.exp(z_tilde)

        solver_end = time.time()
        self.solver_runtime = solver_end - solver_start

        method_end = time.time()
        self.total_runtime = method_end - method_start

        self.depth_map = np.ones_like(data.mask, dtype=np.float) * np.nan
        self.depth_map[data.mask] = z

        # create a mesh from the depth map
        self.facets = construct_facets_from_depth_map_mask(data.mask)
        self.vertices = map_depth_map_to_point_clouds(self.depth_map, data.mask, data.K)
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
    result = PerspectiveDiscretePoisson(data)

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
