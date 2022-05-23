# based on Yvain Queau's Matlab Code
# https://github.com/yqueau/normal_integration/blob/master/Toolbox/mumford_shah_integration.m

import sys
sys.path.append("..")
sys.path.append(".")

from scipy.sparse.linalg import cg
from utils import *
import pyvista as pv
import time
from orthographic_five_point_plane_fitting import OrthographicFivePoint
from scipy.sparse import diags, eye
from tqdm import tqdm
from orthographic_discrete_poisson import generate_dx_dy


class OrthographicMS:
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
    def __init__(self, data, setting):
        self.method_name = "orthographic_Mumford_Shah"
        print("running {}...".format(self.method_name))
        method_start = time.time()
        npix = np.sum(data.mask)

        # Eq. (4) in "Normal Integration: A Survey."
        p = - data.n[data.mask, 0] / data.n[data.mask, 2]
        q = - data.n[data.mask, 1] / data.n[data.mask, 2]

        dvp, dvn, dup, dun = generate_dx_dy(data.mask, data.step_size)
        Duptdup = dup.T @ dup
        Duntdun = dun.T @ dun
        Dvptdvp = dvp.T @ dvp
        Dvntdvn = dvn.T @ dvn

        # Initialization
        z_vec = OrthographicFivePoint(data).depth_map[data.mask]
        bw = (1/(4*setting.epsilon)) * np.ones(npix)
        loss = 0

        # alternating optimization loop
        solver_start = time.time()
        for step in tqdm(range(setting.max_iter)):
            # update w
            Eup2_mat = diags((dup @ z_vec - p) ** 2)
            Eun2_mat = diags((dun @ z_vec - p) ** 2)
            Evp2_mat = diags((dvp @ z_vec - q) ** 2)
            Evn2_mat = diags((dvn @ z_vec - q) ** 2)

            Aup = setting.mu * Eup2_mat + setting.epsilon * Duptdup + (1/(4*setting.epsilon))*eye(npix)
            Avp = setting.mu * Evp2_mat + setting.epsilon * Dvptdvp + (1/(4*setting.epsilon))*eye(npix)
            Aun = setting.mu * Eun2_mat + setting.epsilon * Duntdun + (1/(4*setting.epsilon))*eye(npix)
            Avn = setting.mu * Evn2_mat + setting.epsilon * Dvntdvn + (1/(4*setting.epsilon))*eye(npix)

            wup, _ = cg(Aup, bw, maxiter=500, tol=1e-7)
            wun, _ = cg(Aun, bw, maxiter=500, tol=1e-7)
            wvp, _ = cg(Avp, bw, maxiter=500, tol=1e-7)
            wvn, _ = cg(Avn, bw, maxiter=500, tol=1e-7)

            Wup2_mat = diags(wup ** 2)
            Wun2_mat = diags(wun ** 2)
            Wvp2_mat = diags(wvp ** 2)
            Wvn2_mat = diags(wvn ** 2)

            # update z
            A = setting.mu * (dup.T @ Wup2_mat @ dup +
                              dun.T @ Wun2_mat @ dun +
                              dvp.T @ Wvp2_mat @ dvp +
                              dvn.T @ Wvn2_mat @ dvn)
            b = setting.mu * (dup.T @ Wup2_mat + dun.T @ Wun2_mat) @ p + \
                setting.mu * (dvp.T @ Wvp2_mat + dvn.T @ Wvn2_mat) @ q

            z_vec, _ = cg(A, b, maxiter=500, tol=1e-7)

            def norm(x):
                return np.sqrt(np.sum(x**2))

            loss_old = loss
            loss = norm((wup - 1) ** 2)/(4 * setting.epsilon) + \
                   norm((wup - 1) ** 2)/(4 * setting.epsilon) + \
                   norm((wup - 1) ** 2)/(4 * setting.epsilon) + \
                   norm((wup - 1) ** 2)/(4 * setting.epsilon) + \
                   setting.epsilon * norm(dup * wup) + \
                   setting.epsilon * norm(dun * wun) + \
                   setting.epsilon * norm(dvp * wvp) + \
                   setting.epsilon * norm(dvn * wvn) + \
                   setting.mu * norm(np.sqrt(Eup2_mat) * wup) + \
                   setting.mu * norm(np.sqrt(Eun2_mat) * wun) + \
                   setting.mu * norm(np.sqrt(Evp2_mat) * wvp) + \
                   setting.mu * norm(np.sqrt(Evn2_mat) * wvn)

            relative_loss = np.abs(loss - loss_old) / loss_old
            if relative_loss < 1e-5:
                break
            print(f"step {step} loss {loss}")

        solver_end = time.time()

        self.solver_runtime = solver_end - solver_start
        self.total_runtime = solver_end - method_start
        self.depth_map = np.ones_like(data.mask, dtype=float) * np.nan
        self.depth_map[data.mask] = z_vec

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
    parser.add_argument('--mu', type=float, default=45)
    par = parser.parse_args()

    class Setting:
        pass

    setting = Setting()
    setting.max_iter = 300
    setting.mu = par.mu
    setting.epsilon = 0.01

    data = data_loader(par.path)
    result = OrthographicMS(data)

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


















