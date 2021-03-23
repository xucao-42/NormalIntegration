import pyvista as pv
pv.set_plot_theme("doc")
from glob import glob
import os
from data_anisotropic_gaussian import anisotropic_gaussian_generator
from data_vase import vase_generator
from data_sphere import sphere_orth_generator
import time

objs = [sphere_orth_generator(128),
        vase_generator(128),
        anisotropic_gaussian_generator(150)]

ply_dirs = ["results/2020_11_20_17_47", "results/2020_11_20_17_54", "results/2020_11_20_17_55"]
draw_time = str(time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time())))


fnames = ["vase"]

for fname in fnames:
    for obj in objs:
        if obj.fname == fname:
            data = obj
            break
    surfs = [data.surf]
    c = surfs[0].plot(diffuse=0.5,
                      ambient=0.5,
                      specular=0.3,
                      color="w",
                      smooth_shading=False,
                      show_scalar_bar=False,
                      show_axes=False,
                      eye_dome_lighting=True,
                      # show_edges=True,
                      off_screen=False)

    for ply_dir in ply_dirs:
        ply_list = glob(os.path.join(ply_dir, fname,  "*_depth.ply"))
        ply_list.sort()

        os.makedirs(os.path.join(ply_dir, fname, "img_" + draw_time))
        surfs = [data.surf]
        surfs += [pv.read(i) for i in ply_list]
        fname_list = ["gt"]
        fname_list += [i.split("/")[-1][:-4] for i in ply_list]

        # error_map_list = glob(os.path.join(ply_dir, fname, "{}*.png".format(fname)))
        # error_map_list += glob(os.path.join(ply_dir, fname, "*rmse*.png".format(fname)))
        # error_map_list += glob(os.path.join(ply_dir, fname, "*side*.png".format(fname)))
        #
        # error_map_list += glob(os.path.join(ply_dir, "*rmse*.png"))
        from utils import crop_a_set_of_images
        # crop_a_set_of_images(*error_map_list)


        for idx, surf in enumerate(surfs):
            surf.plot(cpos=c,
                      diffuse=0.5,
                      ambient=0.5,
                      specular=0.3,
                      color="w",
                      smooth_shading=False,
                      show_scalar_bar=False,
                      show_axes=False,
                      eye_dome_lighting=True,
                      # show_edges=True,
                      off_screen=True,
                      screenshot=os.path.join(ply_dir, fname,"img_" + draw_time,  fname_list[idx] + ".png"),
                      window_size=(1024, 768))

        type = "ort"
        ply_img_list = glob(os.path.join(ply_dir, fname, "img_" + draw_time,  "{0}*{1}*.png".format(type, fname)))
        ply_img_list.append(os.path.join(ply_dir, fname, "img_" + draw_time, "gt.png"))
        crop_a_set_of_images(*ply_img_list)
