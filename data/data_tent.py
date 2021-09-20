from data.data_class import Data
from utils import *

def tent_generator(H, slope=1, bound=0.6):
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
    YY = np.flip(YY, axis=0)  # XX points rightwards, YY points upwards

    z = np.zeros_like(XX)
    zx = np.zeros_like(XX)
    zy = np.zeros_like(XX)

    mask_top = np.logical_and.reduce((XX > -bound, XX < bound, YY > 0, YY < bound))
    zy[mask_top] = -slope
    z[mask_top] = bound * slope - slope * YY[mask_top]

    mask_bottom = np.logical_and.reduce((XX > -bound, XX < bound, YY < 0, YY > -bound))
    zy[mask_bottom] = slope
    z[mask_bottom] = bound * slope + slope * YY[mask_bottom]

    data.mask = np.ones((H, H), bool)
    data.depth_gt = -z

    data.p = zx
    data.q = zy

    n = normalize_normal_map(np.stack((-zx, -zy, np.ones_like(zx)), axis=-1))
    data.n = camera_to_object(n)
    data.n[~data.mask] = np.nan

    data.n_vis = (n + 1) / 2
    data.n_vis[~data.mask] = 1

    data.vertices = np.zeros_like(data.n)
    data.vertices[..., 0] = YY + 1
    data.vertices[..., 1] = XX + 1
    data.vertices[..., 2] = data.depth_gt
    data.vertices = data.vertices[data.mask]
    data.fname = "tent"
    data.projection = "orthographic"
    data.construct_mesh()
    return data
