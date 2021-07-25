from data.data_class import Data
from utils import *


def vase_generator(H):
    #     y
    #     |
    #     |                 x  z
    #    / ------ x         | /
    #   /                   |/
    # z                     o------y
    # left: Object coordinate (where we generate the height map and VISUALIZE the normal map)
    # right: Camera coordinate (where the surface / normal map located and where we perform normal integration)
    data = Data()
    x, data.step_size = np.linspace(-6.4, 6.4, num=H, retstep=True)
    y = np.linspace(-6.4, 6.4, num=H)
    XX, YY = np.meshgrid(x, y)
    YY = np.flip(YY, axis=0)  # XX points rightwards, YY points upwards

    Y_hat = YY / 12.8

    py = -138.24 * Y_hat ** 6 \
         + 92.16 * Y_hat ** 5 \
         + 84.48 * Y_hat ** 4 \
         - 48.64 * Y_hat ** 3 \
         - 17.60 * Y_hat ** 2 \
         + 6.40 * Y_hat ** 1 \
         + 3.20

    data.mask = (py ** 2 - XX ** 2) > 0.03
    z = np.sqrt(py ** 2 - XX ** 2)
    data.depth_gt = -z

    zx = - XX / z
    py_y = (-138.24 * 6 * Y_hat ** 5 \
            + 92.16 * 5 * Y_hat ** 4 \
            + 84.48 * 4 * Y_hat ** 3 \
            - 48.64 * 3 * Y_hat ** 2 \
            - 17.60 * 2 * Y_hat ** 1 \
            + 6.40) / 12.8
    zy = (py / z) * py_y
    zx[~data.mask], zy[~data.mask] = 0, 0

    n = normalize_normal_map(np.stack((-zx, -zy, np.ones_like(zx)), axis=-1))

    data.n = camera_to_object(n)
    data.n[~data.mask] = np.nan

    data.n_vis = (n + 1) / 2
    data.n_vis[~data.mask] = 1

    data.vertices = np.zeros_like(data.n)
    data.vertices[..., 0] = YY + 6.4
    data.vertices[..., 1] = XX + 6.4
    data.vertices[..., 2] = data.depth_gt
    data.vertices = data.vertices[data.mask]
    data.fname = "vase"
    data.projection = "orthographic"
    data.construct_mesh()
    return data