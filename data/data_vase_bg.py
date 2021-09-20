from data.data_class import Data
from utils import *

def vase_bg_generator(H, padding=20):
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
    # YY = np.flip(YY, axis=0)  # XX points rightwards, YY points upwards

    Y_hat = YY / 12.8

    py = -138.24 * Y_hat ** 6 \
         + 92.16 * Y_hat ** 5 \
         + 84.48 * Y_hat ** 4 \
         - 48.64 * Y_hat ** 3 \
         - 17.60 * Y_hat ** 2 \
         + 6.40 * Y_hat ** 1 \
         + 3.20

    mask = (py ** 2 - XX ** 2) > 0.03
    z = np.sqrt(py ** 2 - XX ** 2)
    z[~mask] = 0

    zx = - XX / z
    py_y = (-138.24 * 6 * Y_hat ** 5 \
            + 92.16 * 5 * Y_hat ** 4 \
            + 84.48 * 4 * Y_hat ** 3 \
            - 48.64 * 3 * Y_hat ** 2 \
            - 17.60 * 2 * Y_hat ** 1 \
            + 6.40) / 12.8
    zy = (py / z) * py_y
    zx[~mask], zy[~mask] = 0, 0


    n = normalize_normal_map(np.stack((zx, -zy, np.ones_like(zx)), axis=-1))
    n = camera_to_object(n)

    data.mask = np.ones((H+padding*2, H+padding*2), bool)
    data.n = np.zeros((H+padding*2, H+padding*2, 3), float)
    data.n[..., -1] = -1
    data.n[padding:H+padding, padding:H+padding, :] = n

    data.p = - data.n[..., 0] / data.n[..., 2]
    data.q = - data.n[..., 1] / data.n[..., 2]

    data.n_vis = (camera_to_object(data.n) + 1) / 2

    data.depth_gt = np.zeros((H+padding*2, H+padding*2), float)
    data.depth_gt[padding:H+padding, padding:H+padding] = z

    x = np.arange(H+padding*2) * data.step_size
    y = np.arange(H+padding*2) * data.step_size
    XX, YY = np.meshgrid(x, y)
    YY = np.flip(YY, axis=0)  # XX points rightwards, YY points upwards

    data.vertices = np.zeros_like(data.n)
    data.vertices[..., 0] = YY
    data.vertices[..., 1] = XX
    data.vertices[..., 2] = data.depth_gt
    data.vertices = data.vertices[data.mask]
    data.fname = "vase_bg"
    data.projection = "orthographic"
    data.construct_mesh()
    return data
