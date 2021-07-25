from data.data_class import Data
import numpy as np
import matplotlib.pyplot as plt
from utils import normalize_normal_map, camera_to_object, construct_facets_from_depth_map_mask
import pyvista as pv


def anisotropic_gaussian_generator(H=150):
    #     y
    #     |
    #     |                 x  z
    #    / ------ x         | /
    #   /                   |/
    # z                     o------y
    # left: Object coordinate (where we generate the height map and VISUALIZE the normal map)
    # right: Camera coordinate (where the surface / normal map located and where we perform normal integration)
    data = Data()
    x, data.step_size = np.linspace(-1, 10, num=H, retstep=True)
    y = np.linspace(-1, 10, num=H)
    XX, YY = np.meshgrid(x, y)
    YY = np.flip(YY, axis=0)   # XX points rightwards, YY points upwards

    A = [2.5, 3, -5, -2, 5]
    p = np.array([[1, 2],
                  [7, 4],
                  [5, 5],
                  [2, 8],
                  [6, 8]])
    LAMBDA = np.array([[[3, -1], [-1, 3]],
                       [[2, -1], [-1, 4]],
                       [[2, 1], [1, 5]],
                       [[5, 1], [1, 3]],
                       [[4, -1], [-1, 1]]])
    LAMBDA = np.linalg.inv(LAMBDA)

    z = np.zeros((H, H), dtype=np.float)
    zx = np.zeros((H, H), dtype=np.float)
    zy = np.zeros((H, H), dtype=np.float)
    for i in range(5):
        exp = LAMBDA[i, 0, 0] * (XX - p[i][0]) ** 2 + 2 * LAMBDA[i, 0, 1] * (XX - p[i][0]) * (YY - p[i][1]) + LAMBDA[i, 1, 1] * (YY - p[i][1]) ** 2
        z += A[i] * np.exp(-0.5 * exp)
        zxx = -0.5 * (2 * LAMBDA[i, 0, 0] * (XX - p[i][0]) + 2 * LAMBDA[i, 0, 1] * (YY - p[i][1]))
        zyy = -0.5 * (2 * LAMBDA[i, 1, 1] * (YY - p[i][1]) + 2 * LAMBDA[i, 0, 1] * (XX - p[i][0]))
        zx += A[i] * np.exp(-0.5 * exp) * zxx
        zy += A[i] * np.exp(-0.5 * exp) * zyy

    n = normalize_normal_map(np.stack((-zx, -zy, np.ones_like(zx)), axis=-1))

    data.n = camera_to_object(n)
    data.n_vis = (n + 1) / 2
    data.depth_gt = -z

    data.mask = np.ones((H, H), dtype=np.bool)
    data.facets = construct_facets_from_depth_map_mask(data.mask)
    data.vertices = np.concatenate([(YY.flatten()[..., None] + 1),
                                    (XX.flatten()[..., None] + 1),
                                    data.depth_gt.flatten()[..., None]], axis=-1)
    data.surf = pv.PolyData(data.vertices, data.facets)
    data.fname = "anisotropic_gaussian"
    return data
