import time
import os
import numpy as np
import cv2
import argparse
import sys
sys.path.append("methods")
sys.path.append("data")

from methods.orthographic_discrete_poisson import OrthographicPoisson
from methods.orthographic_five_point_plane_fitting import OrthographicFivePoint
from methods.orthographic_four_point_plane_fitting import OrthographicFourPoint
from methods.orthographic_DGP import OrthographicDiscreteGeometryProcessing
from methods.orthographic_discrete_functional import OrthographicDiscreteFunctional

from data.data_sphere import sphere_orth_generator
from data.data_vase import vase_generator
from data.data_anisotropic_gaussian import anisotropic_gaussian_generator


parser = argparse.ArgumentParser()
parser.add_argument('-n', "--noise", type=float, default=0, help='the standard deviation of Gaussian noise added to the input normal map (default 0)')
parser.add_argument('-o', "--outlier", type=float, default=0, help='the percentage of outliers (range from 0 to 1) added to the input normal map (default 0)')
par = parser.parse_args()

str_time = str(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())))

objs = [
        sphere_orth_generator(128),
        vase_generator(128),
        anisotropic_gaussian_generator(150)
        ]

for obj in objs:
    result_dir = os.path.join("results",  str_time, obj.fname)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if par.noise and par.outlier:
        obj.add_outlier_on_noise_map(par.outlier, par.noise)
        obj.input_normal_map = obj.n_outlier_noise.copy()
    elif par.noise:
        obj.add_noise(par.noise)
        obj.input_normal_map = obj.n_noise.copy()
    elif par.outlier:
        obj.add_outlier(par.outlier)
        obj.input_normal_map = obj.n_outlier.copy()
    else:
        obj.input_normal_map = obj.n.copy()

    obj.n = obj.input_normal_map.copy()
    from utils import camera_to_object
    n_vis = (camera_to_object(obj.n) + 1)/2
    n_vis[~obj.mask] = 1
    cv2.imwrite(os.path.join(result_dir, "input_normal_map.png"),
                cv2.cvtColor((n_vis * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

    results = [
                OrthographicFivePoint(obj),
                OrthographicFourPoint(obj),
                OrthographicPoisson(obj),
                OrthographicDiscreteFunctional(obj),
                OrthographicDiscreteGeometryProcessing(obj)
              ]

    absolute_difference_maps = []
    rmse_list = []
    mae_list = []
    std_list = []
    method_name_list = []
    solver_runtime_list = []
    total_runtime_list = []
    for z_est in results:
        method_name_list.append(z_est.method_name)
        solver_runtime_list.append(z_est.solver_runtime)
        total_runtime_list.append(z_est.total_runtime)

        offset = np.nanmean(obj.depth_gt - z_est.depth_map)
        offset_depth = z_est.depth_map + offset
        absolute_difference_map = np.abs(offset_depth - obj.depth_gt)
        rmse = np.sqrt(np.nanmean(absolute_difference_map ** 2))
        absolute_difference_maps.append(absolute_difference_map)
        rmse_list.append(rmse)
        mae_list.append(np.nanmean(absolute_difference_map))
        std_list.append(np.nanstd(absolute_difference_map))

        z_est.surface.points[:, 2] += offset
        z_est.surface.save(os.path.join(result_dir, "mesh_{}.ply".format(z_est.method_name)), binary=True)
    obj.surf.save(os.path.join(result_dir, "mesh_gt.ply"), binary=True)

    from utils import apply_jet_on_multiple_error_maps

    absolute_difference_maps_JET = apply_jet_on_multiple_error_maps(absolute_difference_maps, sigma_multiplier=2)
    for idx, abs_diff_map_jet in enumerate(absolute_difference_maps_JET):
        cv2.imwrite(filename=os.path.join(result_dir, "absolute_difference_map_{}.png".format(results[idx].method_name)),
                    img=abs_diff_map_jet)

    import csv
    with open(os.path.join(result_dir, 'evaluation_metric_summary.csv'), 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["metric"] + method_name_list)
        writer.writerow(["RMSE"] + rmse_list)
        writer.writerow(["MAbsE"] + mae_list)
        writer.writerow(["std"] + std_list)
        writer.writerow(["solver_time (sec)"] + solver_runtime_list)
        writer.writerow(["total_runtime (sec)"] + total_runtime_list)

    from glob import glob
    crop_img_list = glob(os.path.join(result_dir, "absolute_difference_map*.png"))
    crop_img_list += glob(os.path.join(result_dir, "input_normal_map.png"))

    from utils import crop_a_set_of_images
    try:
        crop_a_set_of_images(*crop_img_list)
    except:
        pass

