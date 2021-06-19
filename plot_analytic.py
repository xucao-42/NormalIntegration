import pyvista as pv
pv.set_plot_theme("doc")
from glob import glob
import os
import time
from utils import mkdir

ply_dir = os.path.join("results", "2021_04_08_17_56", "reading")

ply_list = glob(os.path.join(ply_dir, "*.ply"))
surfaces = [pv.read(i) for i in ply_list]

draw_time = "draw_time_" + str(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())))
save_dir = os.path.join(ply_dir, draw_time)
mkdir(save_dir)

camera_pose = surfaces[0].plot()

for idx, surf in enumerate(surfaces):
    fname = ply_list[idx].split("/")[-1].split(".")[0]
    surf.plot(cpos=camera_pose,
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
              screenshot=os.path.join(save_dir, "{}.png".format(fname)),
              window_size=(1024, 768))

ply_img_list = glob(os.path.join(save_dir, "*.png"))
from utils import crop_a_set_of_images
crop_a_set_of_images(*ply_img_list)
