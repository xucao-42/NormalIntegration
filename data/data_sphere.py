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

    n = normalize_normal_map(np.stack((-zx, -zy, np.ones_like(zx)), axis=-1))

    data.n = camera_to_object(n)
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

    return data
