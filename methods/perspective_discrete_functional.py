import sys
sys.path.append("..")
sys.path.append(".")

import numpy as np
from scipy.sparse.linalg import lsqr, cg
from utils import construct_facets_from_depth_map_mask, map_depth_map_to_point_clouds
import pyvista as pv
from orthographic_discrete_functional import generate_dx_dy_wb
from scipy.sparse import vstack
import time, os

class PerspectiveDiscreteFunctional:
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
        self.method_name = "perspective_discrete_functional"
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

        b = np.concatenate((p_tilde[data.mask], q_tilde[data.mask]))
        Du, Dv = generate_dx_dy_wb(data.mask)
        A = vstack((Du, Dv))

        solver_start = time.time()

        # z_tilde = lsqr(A, b)[0]
        z_tilde, _ = cg(A.T @ A, A.T @ b, maxiter=1000, tol=1e-9)

        # Eq. (13) in "Normal Integration: A Survey."
        z = np.exp(z_tilde)

        solver_end = time.time()
        self.solver_runtime = solver_end - solver_start

        method_end = time.time()
        self.total_runtime = method_end - method_start

        self.depth_map = np.ones_like(data.mask, dtype=np.float) * np.nan
        self.depth_map[data.mask] = z

        # construct a mesh from the depth map
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
    result = PerspectiveDiscreteFunctional(data)

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

