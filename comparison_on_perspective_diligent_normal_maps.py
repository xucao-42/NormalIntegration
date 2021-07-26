import time, os
import numpy as np
from itertools import product
import cv2
import sys
sys.path.append("methods")
sys.path.append("data")

from methods.perspective_four_point_plane_fitting import PerspectiveFourPointPlaneFitting
from methods.perspective_five_point_plane_fitting import PerspectiveFivePointPlaneFitting
from methods.perspective_zhu_and_smith_cd import PerspectiveZhuCD
from methods.perspective_discrete_functional import PerspectiveDiscreteFunctional
from methods.perspective_discrete_poisson import PerspectiveDiscretePoisson
from methods.perspective_zhu_and_smith_sg import PerspectiveZhuSG

from data_diligent import DataDiligent

class Setting:
    pass


surface_name = [
            "bear",
            "buddha",
            "cat",
            "cow",
            "goblet",
            "harvest",
            "pot1",
            "pot2",
            "reading"
            ]

method_type = [
        "gt",
        # "ECCV2020",
        # "l2",
        ]

setting = Setting()
st_time = str(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())))

for objname in product(surface_name, method_type):
    print(objname)
    if objname[1] != "gt":  # if gt normal is used as input, exclude boundary points
        obj = DataDiligent(*objname, exclude_bouday=0)
    else:
        obj = DataDiligent(*objname, exclude_bouday=1)

    setting.save_dir = os.path.join("results", st_time, objname[0] + "_" + objname[1])
    if not os.path.exists(setting.save_dir):
        os.makedirs(setting.save_dir)
    obj.save_n(setting.save_dir)

    setting.polynomial_order = 3
    setting.num_neighbor = 25
    setting.lambda_smooth = 1

    results = [
               PerspectiveFourPointPlaneFitting(obj),
               PerspectiveFivePointPlaneFitting(obj),
               PerspectiveDiscreteFunctional(obj),
               PerspectiveDiscretePoisson(obj),
               PerspectiveZhuCD(obj, setting),
               PerspectiveZhuSG(obj, setting)
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

        scale = np.nanmean(obj.depth_gt / z_est.depth_map)
        scaled_depth = z_est.depth_map * scale
        absolute_difference_map = np.abs(scaled_depth - obj.depth_gt)
        rmse = np.sqrt(np.nanmean(absolute_difference_map ** 2))
        absolute_difference_maps.append(absolute_difference_map)
        rmse_list.append(rmse)
        mae_list.append(np.nanmean(absolute_difference_map))
        std_list.append(np.nanstd(absolute_difference_map))

        z_est.surface.points *= scale
        z_est.surface.save(os.path.join(setting.save_dir, "mesh_{}.ply".format(z_est.method_name)), binary=True)
    obj.surf.save(os.path.join(setting.save_dir, "mesh_gt.ply"), binary=True)

    from utils import apply_jet_on_multiple_error_maps

    absolute_difference_maps_JET = apply_jet_on_multiple_error_maps(absolute_difference_maps, sigma_multiplier=2)
    for idx, abs_diff_map_jet in enumerate(absolute_difference_maps_JET):
        cv2.imwrite(filename=os.path.join(setting.save_dir, "absolute_difference_map_{}.png".format(results[idx].method_name)),
                    img=abs_diff_map_jet)

    import csv

    with open(os.path.join(setting.save_dir, 'evaluation_metric_summary.csv'), 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["metric"] + method_name_list)
        writer.writerow(["RMSE"] + rmse_list)
        writer.writerow(["MAbsE"] + mae_list)
        writer.writerow(["std"] + std_list)
        writer.writerow(["solver_time (sec)"] + solver_runtime_list)
        writer.writerow(["total_runtime (sec)"] + total_runtime_list)

    from glob import glob

    crop_img_list = glob(os.path.join(setting.save_dir, "absolute_difference_map*.png"))
    crop_img_list += glob(os.path.join(setting.save_dir, "input_normal_map.png"))

    from utils import crop_a_set_of_images

    try:
        crop_a_set_of_images(*crop_img_list)
    except:
        pass
