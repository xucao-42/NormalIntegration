from data.data_sphere import sphere_orth_generator
from data.data_vase import vase_generator
from data.data_anisotropic_gaussian import anisotropic_gaussian_generator

from methods.orthographic_poisson import PoissonOrthographic
from methods.orthographic_five_point_fitting import FivePointOthographic
from methods.orthographic_four_point_plane_fitting import FourPointOrthographic
from methods.orthographic_DGP import DiscreteGeometryProcessingOrthographic
from methods.orthographic_discrete_functional import DiscreteFunctionalOrthographic

import time, os
import numpy as np

class Setting:
    pass

setting = Setting()
st_time = str(time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time())))

objs = [sphere_orth_generator(128),
        vase_generator(128),
        anisotropic_gaussian_generator(150)]

for obj in objs:
    setting.save_dir = os.path.join("results",  st_time, obj.fname)
    if not os.path.exists(setting.save_dir):
        os.makedirs(setting.save_dir)

    setting.use_bg = False
    setting.add_noise = 0
    setting.add_outlier = 0

    if setting.add_noise and setting.add_outlier:
        obj.n_used = obj.n_outlier_noise.copy()
    elif setting.add_noise:
        obj.n_used = obj.n_noise.copy()
    elif setting.add_outlier:
        obj.n_used = obj.n_outlier.copy()
    else:
        obj.n_used = obj.n.copy()

    obj.save_n(setting.save_dir, use_nosie=setting.add_noise, use_outlier=setting.add_outlier)

    results = [FivePointOthographic(obj, setting),
               FourPointOrthographic(obj, setting),
                PoissonOrthographic(obj, setting),
                DiscreteFunctionalOrthographic(obj, setting),
                DiscreteGeometryProcessingOrthographic(obj, setting)]

    rmse_maps = []
    rmse_list = []
    mae_list = []
    std_list = []
    for z_est in results:
        offset = np.nanmean(obj.depth_gt - z_est.depth)
        offset_depth = z_est.depth + offset
        rmse_map = np.abs(offset_depth - obj.depth_gt)
        rmse = np.sqrt(np.nanmean(rmse_map ** 2))
        rmse_maps.append(rmse_map)
        rmse_list.append(rmse)
        mae_list.append(np.nanmean(rmse_map))
        std_list.append(np.nanstd(rmse_map))

        z_est.surf.points[:, 2] += offset
        z_est.surf.save(os.path.join(setting.save_dir, z_est.method_name + "_" + obj.fname + "_depth.ply"), binary=False)

    np.save(os.path.join(setting.save_dir, "ad_maps"), np.array(rmse_maps))
    mu = np.nanmean(np.array(rmse_maps))
    sigma = np.nanstd(np.array(rmse_maps))

    import cv2
    for idx, z_est in enumerate(results):
        rmse_map = rmse_maps[idx] / (mu + 2 * sigma)
        rmse_map[rmse_map > 1] = 1
        nan_mask = np.isnan(rmse_map)
        rmse_map[nan_mask] = 1
        rmse_jet = cv2.applyColorMap((255 * rmse_map).astype(np.uint8), cv2.COLORMAP_JET)
        rmse_jet[nan_mask] = 255
        cv2.imwrite(filename=os.path.join(setting.save_dir, 
                    z_est.method_name + "_rmse_{0:.5e}_mae_{1:.5e}_std_{2:.5e}".format(rmse_list[idx], mae_list[idx], std_list[idx]).replace(".", "_") + ".png"),
                    img=rmse_jet)

    from glob import glob
    error_map_list = glob(os.path.join(setting.save_dir, "{}*.png".format(obj.fname)))
    error_map_list += glob(os.path.join(setting.save_dir, "*rmse*.png".format(obj.fname)))

    from utils import crop_a_set_of_images
    try:
        crop_a_set_of_images(*error_map_list)
    except:
        pass

