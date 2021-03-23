from data.data_class import Data
from utils import *

def sphere_orth_generator(H=128):
    #     y
    #     |
    #     |                 x  z
    #    / ------ x         | /
    #   /                   |/
    # z                     o------y
    # left: Object coordinate (where we generate the height map and VISUALIZE the normal map)
    # right: Camera coordinate (where the surface / normal map located and where we perform normal integration)
    data = Data()
    x, data.step_size = np.linspace(-1, 1, num=H, retstep=True)
    y = np.linspace(-1, 1, num=H)
    XX, YY = np.meshgrid(x, y)
    YY = np.flip(YY, axis=0)   # XX points rightwards, YY points upwards

    data.mask = (1 - XX ** 2 - YY ** 2) > 1e-7
    z = np.sqrt(1 - XX ** 2 - YY ** 2)

    zx = - XX / z
    zy = - YY / z
    zx[~data.mask], zy[~data.mask] = 0, 0

    n = normalize_normal_map(np.concatenate((-zx[..., None],
                        -zy[..., None],
                        np.ones_like(zx)[..., None]), axis=-1))
    data.n = world_to_object(n)
    data.n_vis = (n + 1) / 2
    data.n_vis[~data.mask] = 1
    data.depth_gt = -z

    data.vertices = np.zeros_like(n, dtype=np.float)
    data.vertices[..., 0] = YY + 1
    data.vertices[..., 1] = XX + 1
    data.vertices[..., 2] = data.depth_gt
    data.vertices = data.vertices[data.mask]
    data.construct_mesh()

    data.projection = "orthographic"
    data.fname = "sphere"

    data.add_noise()
    data.add_background(pad_width=0)
    data.add_outlier()
    data.add_outlier_on_noise_map(0.05)

    return data

# sphere_o.depth_gt_wbg = np.pad(sphere_o.depth_gt, ((10, 10), (10, 10)), "constant", constant_values=0)
# sphere_o.bg_only_mask = np.pad(sphere_o.bg_only_mask, ((10, 10), (10, 10)), "constant", constant_values=1)
# sphere_o.fg_mask = np.pad(sphere_o.mask, ((10, 10), (10, 10)), "constant", constant_values=0)
# sphere_o.construct_mesh()
# sphere_o.add_outlier()
# sphere_o.add_outlier_on_noise_map(0.1)
#
# yy, xx = np.meshgrid(range(H+20), range(H+20))
# xx = np.max(xx) - xx
# v_0 = np.zeros((H+20, H+20, 3))
# v_0[..., 0] = xx
# v_0[..., 1] = yy
# v_0[..., 2] = sphere_o.depth_gt_wbg
#
# sphere_o.vertices_bg = v_0.reshape(-1, 3)
# sphere_o.facets_bg = construct_facet_for_depth(sphere_o.bg_mask)
# sphere_o.surf_bg = pv.PolyData(sphere_o.vertices_bg, sphere_o.facets_bg)

# nz = sphere_o.n[sphere_o.mask, 2]
# angle = np.rad2deg(np.arcsin(-nz))
# plt.hist(angle, bins=90)
# plt.show()