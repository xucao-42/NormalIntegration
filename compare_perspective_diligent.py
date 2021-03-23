# from data import sphere as obj
import matplotlib.pyplot as plt
import time, os
from methods.perspective_four_point import FourPointPlaneFittingPerspective
from methods.perspective_five_point import FivePointPlaneFittingPerspective
from methods.perspective_zhu_and_smith_cd import ZhuCDPerspective
from methods.perspective_discrete_functional import DiscreteFunctionalPerspective
from methods.perspective_discrete_poisson import DiscretePoissonPerspective
from methods.perspective_zhu_and_smith_sg import ZhuSGPerspective

import numpy as np
from itertools import product
from data_diligent import DataDiligent

class Setting:
    pass

obj_name = ["bear",
            "buddha",
            "cat",
            "cow",
            "goblet",
            "harvest",
            "pot1",
            "pot2",
            "reading"]

type = ["ECCV2020"]

setting = Setting()
st_time = str(time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time())))
for objname in product(obj_name, type):
    print(objname)
    if objname[1] != "gt":  # if gt normal is used, exclued boundary
        obj = DataDiligent(*objname, exclude_bouday=0)
    else:
        obj = DataDiligent(*objname, exclude_bouday=1)

    setting.save_dir = os.path.join("results", st_time, objname[0])
    setting.add_outlier = 0
    setting.add_noise = 0
    if not os.path.exists(setting.save_dir):
        os.makedirs(setting.save_dir)
    obj.save_n(setting.save_dir)

    setting.polynomial_order = 3
    setting.num_neighbor = 25
    setting.lambda_smooth = 0

    results = [FourPointPlaneFittingPerspective(obj, setting),
               FivePointPlaneFittingPerspective(obj, setting),
               DiscreteFunctionalPerspective(obj, setting),
               ZhuCDPerspective(obj, setting),
               DiscretePoissonPerspective(obj, setting),
               ZhuSGPerspective(obj, setting)]

    rmse_maps = []
    rmse_list = []
    ratio_maps = []
    SIDE_list = []
    mae_list = []
    for z_est in results:
        scale = np.nanmean(obj.depth_gt / z_est.depth)
        scaled_depth = z_est.depth * scale
        rmse_map = np.abs(scaled_depth - obj.depth_gt)
        rmse = np.sqrt(np.nanmean(rmse_map ** 2))
        rmse_maps.append(rmse_map)
        rmse_list.append(rmse)

        z_est.surf.points *= scale
        z_est.surf.save(os.path.join(setting.save_dir, z_est.method + "_" + obj.fname + "_depth.ply"), binary=False)
        mae_list.append(np.nanmean(rmse_map))

        ratio_map = np.log(obj.depth_gt / z_est.depth)
        ratio_map -= np.nanmean(ratio_map)
        SIDE_list.append(np.nanstd(ratio_map))
        ratio_maps.append(np.abs(ratio_map))

    mu = np.nanmean(np.array(rmse_maps))
    sigma = np.nanstd(np.array(rmse_maps))
    mu_side = np.nanmean(np.array(SIDE_list))
    sigma_side = np.nanstd(np.array(SIDE_list))

    import cv2
    for idx, z_est in enumerate(results):
        rmse_map = rmse_maps[idx] / (mu+3*sigma)
        rmse_map[rmse_map > 1] = 1
        nan_mask = np.isnan(rmse_map)
        rmse_map[nan_mask] = 1
        rmse_jet = cv2.applyColorMap((255 * rmse_map).astype(np.uint8), cv2.COLORMAP_JET)
        rmse_jet[nan_mask] = 255
        cv2.imwrite(filename=os.path.join(setting.save_dir,
                    z_est.method + "_rmse_{0:.5e}_mae_{1:.5e}".format(rmse_list[idx], mae_list[idx]).replace(".", "_") + ".png"),
                    img=rmse_jet)

    from glob import glob
    error_map_list = glob(os.path.join(setting.save_dir, "{0}_{1}.png".format(objname[0], objname[1])))
    error_map_list += glob(os.path.join(setting.save_dir, "*rmse*.png"))

    from utils import crop_a_set_of_images
    crop_a_set_of_images(*error_map_list)
