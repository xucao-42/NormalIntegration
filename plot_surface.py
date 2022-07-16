import pyvista as pv
pv.set_plot_theme("document")
from glob import glob
import os
import time
from utils import mkdir
import argparse


if __name__ == "__main__":
    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise NotADirectoryError(string)

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=dir_path)
    parser.add_argument('-s', '--show_edge', type=bool, default=False)

    par = parser.parse_args()

    ply_list = glob(os.path.join(par.path, "*.ply"))
    surfaces = [pv.read(i) for i in ply_list]

    draw_time = "draw_time_" + str(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())))
    save_dir = os.path.join(par.path, draw_time)
    mkdir(save_dir)

    camera_pose = surfaces[0].plot()

    for idx, surf in enumerate(surfaces):
        fname = os.path.basename(ply_list[idx]).split(".")[0]
        surf.plot(cpos=camera_pose,
                  diffuse=0.5,
                  ambient=0.5,
                  specular=0.3,
                  color="w",
                  smooth_shading=False,
                  show_scalar_bar=False,
                  show_axes=False,
                  eye_dome_lighting=True,
                  show_edges=par.show_edge,
                  off_screen=True,
                  screenshot=os.path.join(save_dir, "{}.png".format(fname)),
                  window_size=(1024, 768))

    ply_img_list = glob(os.path.join(save_dir, "*.png"))
    from utils import crop_a_set_of_images
    crop_a_set_of_images(*ply_img_list)
