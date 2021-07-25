import numpy as np
from utils import normalize_normal_map, camera_to_object, curl_of_normal_map
from copy import copy
import cv2
import os
import pyvista as pv
import matplotlib.pyplot as plt
from utils import construct_facets_from_depth_map_mask, crop_image_by_mask

def perspective_sphere_normal_and_depth(H, K, r, d):
    K_1 = np.linalg.inv(K)
    # create homogenouse coordinate
    yy, xx = np.meshgrid(range(H), range(H))
    xx = np.max(xx) - xx
    v_0 = np.zeros((H, H, 3))
    v_0[..., 0] = xx
    v_0[..., 1] = yy
    v_0[..., 2] = 1
    v_0 = v_0.reshape(-1, 3).T
    v_0_3d = (K_1 @ v_0).T.reshape((H, H, 3))

    a = np.sum(v_0_3d ** 2, axis=-1)
    b = -2 * d * v_0_3d[..., 2]
    c = d ** 2 - r ** 2
    mask = b ** 2 - 4 * a * c > 0
    # for i in range(2):
    #     mask = boundary_excluded_mask(mask)

    t = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    v = v_0_3d * t[..., None]
    n = v.copy()
    n[..., 2] -= d
    n[~mask] = 0
    n = normalize_normal_map(n)

    return n, mask, t, v[mask]

def tent_generator(H):
    r = (H - 1) / 2
    n = np.zeros((H, H, 3))
    n[:int(r)+1, ...] = [np.sqrt(2)/2, 0, -np.sqrt(2)/2]
    n[int(r)+1:, ...] = [-np.sqrt(2)/2, 0, -np.sqrt(2)/2]
    n = normalize_normal_map(n)
    return n



def generate_dis_normal_map(d):
    r = (d - 1) / 2
    # II, JJ = np.meshgrid(range(n), range(n))  # x point to right, y point to bottom
    n = np.zeros((d, d, 3))
    n[int(r)+1:, ...] = [0, 0, -1]
    n[:int(r)+1, :int(r)+1, :] = [-np.sqrt(2)/2, 0, -np.sqrt(2)/2]
    # n[:int(r)+1, :int(r)+1, :] = [np.sqrt(2)/2, 0, -1.6]
    n[:int(r)+1, int(r)+1:, :] = [np.sqrt(2)/2, 0, -np.sqrt(2)/2]
    n = normalize_normal_map(n)
    return n


def add_noise(n, mask, std=0.1):
    p = - n[..., 0] / n[..., 2]
    q = - n[..., 1] / n[..., 2]
    # std_noise = std #* np.nanmax(np.sqrt(p ** 2 + q ** 2))
    p += np.random.normal(scale=std, size=p.shape)
    q += np.random.normal(scale=std, size=q.shape)
    n_noise = np.concatenate([p[..., None],
                              q[..., None],
                              -np.ones_like(p)[..., None]], axis=-1)
    n_noise[~mask] = [0, 0, -1]
    n_noise = normalize_normal_map(n_noise)
    return n_noise

class Data:
    def __init__(self):
        self.n = None
        self.mask = None
        self.K = None
        self.mesh_vertice_gt = None
        self.projection = None
        self.fname = None
        self.curl = None
        self.n_vis = None

    def save_n(self, fpath, use_bg=False, use_nosie=False, use_outlier=False):
        if use_bg:
            cv2.imwrite(os.path.join(fpath, self.fname + "_bg.png"), cv2.cvtColor((crop_image_by_mask(self.n_wbg_vis, self.mask) * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        elif use_nosie and use_outlier:
            cv2.imwrite(os.path.join(fpath, self.fname + "_noise_outlier.png"),
                        cv2.cvtColor((crop_image_by_mask(self.n_outlier_noise_vis, self.mask) * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        elif use_nosie:
            cv2.imwrite(os.path.join(fpath, self.fname + "_noise.png"), cv2.cvtColor((crop_image_by_mask(self.n_noise_vis, self.mask) * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        elif use_outlier:
            cv2.imwrite(os.path.join(fpath, self.fname + "_outlier.png"), cv2.cvtColor((crop_image_by_mask(self.n_outlier_vis, self.mask) * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        else:
            cv2.imwrite(os.path.join(fpath, self.fname + ".png"), cv2.cvtColor((crop_image_by_mask(self.n_vis, self.mask) * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

    def add_noise(self, std=0.1):
        self.n_noise = add_noise(self.n, self.mask, std)
        self.n_noise_vis = (camera_to_object(self.n_noise) + 1) / 2
        self.n_noise_vis[~self.mask] = 1

    def add_outlier(self, percentage_outlier=0.01):
        p_map = - self.n[..., 0] / self.n[..., 2]
        q_map = - self.n[..., 1] / self.n[..., 2]
        # randomly choose num_outlier normal vectors and 5 times pq
        num_pixel = np.sum(self.mask)
        image_ij = np.flatnonzero(self.mask)
        idx_list = np.random.choice(image_ij, int(num_pixel * percentage_outlier), replace=False)
        p_map.ravel()[idx_list] *= 5
        q_map.ravel()[idx_list] *= 5
        n_outlier = np.concatenate([p_map[..., None],
                                  q_map[..., None],
                                  -np.ones_like(p_map)[..., None]], axis=-1)
        n_outlier[~self.mask] = [0, 0, -1]
        self.n_outlier = normalize_normal_map(n_outlier)

        self.n_outlier_vis = (camera_to_object(self.n_outlier) + 1) / 2
        self.n_outlier_vis[~self.mask] = 1


    def add_outlier_on_noise_map(self, percentage_outlier=0.01, std_noise=0.1):
        self.add_noise(std_noise)

        p_map = - self.n[..., 0] / self.n[..., 2]
        q_map = - self.n[..., 1] / self.n[..., 2]

        p_map_noise = - self.n_noise[..., 0] / self.n_noise[..., 2]
        q_map_noise = - self.n_noise[..., 1] / self.n_noise[..., 2]
        # randomly choose num_outlier normal vectors and 5 times pq
        num_pixel = np.sum(self.mask)
        image_ij = np.flatnonzero(self.mask)
        idx_list = np.random.choice(image_ij, int(num_pixel * percentage_outlier), replace=False)
        # p_map.ravel()[idx_list] *= 5
        # q_map.ravel()[idx_list] *= 5
        p_map_noise.ravel()[idx_list] = p_map.ravel()[idx_list] * 4
        q_map_noise.ravel()[idx_list] = q_map.ravel()[idx_list] * 4

        n_outlier = np.concatenate([p_map_noise[..., None],
                                    q_map_noise[..., None],
                                    -np.ones_like(p_map)[..., None]], axis=-1)
        n_outlier[~self.mask] = [0, 0, -1]
        self.n_outlier_noise = normalize_normal_map(n_outlier)

        self.n_outlier_noise_vis = (camera_to_object(self.n_outlier_noise) + 1) / 2
        self.n_outlier_noise_vis[~self.mask] = 1



    def add_background(self, pad_width=1):
        x_max, x_min = np.max(np.where(self.mask)[0]), np.min(np.where(self.mask)[0])
        y_max, y_min = np.max(np.where(self.mask)[1]), np.min(np.where(self.mask)[1])
        H, W = self.mask.shape
        if x_max + pad_width <= H and x_min - pad_width >= 0 and y_max + pad_width <= W and y_min - pad_width >=0:
            self.bg_mask = self.mask.copy()
            self.bg_mask[x_min-pad_width: x_max+pad_width+1, y_min-pad_width:y_max+pad_width+1] = True
            pad_area = self.bg_mask ^ self.mask  # xor
            self.n_wbg = self.n.copy()
            self.n_wbg[pad_area] = [0, 0, -1]

        else:
            # if pad area is out of boudary, we first expand the normal map and the mask. Then perform padding as above.
            pad_width_h = pad_width
            pad_width_v = pad_width
            mask_expand = np.pad(self.mask, ((pad_width_v, pad_width_v), (pad_width_h, pad_width_h)), "constant", constant_values=False)
            self.n_wbg = np.zeros((mask_expand.shape[0], mask_expand.shape[1], 3))
            self.n_wbg[mask_expand] = self.n[self.mask]

            self.bg_mask = mask_expand.copy()
            # new object boundary index
            v_max, v_min = np.max(np.where(self.bg_mask)[0]), np.min(np.where(self.bg_mask)[0])
            h_max, h_min = np.max(np.where(self.bg_mask)[1]), np.min(np.where(self.bg_mask)[1])
            self.bg_mask[v_min - pad_width_v: v_max + pad_width_v + 1, h_min - pad_width_h:h_max + pad_width_h+1] = True
            pad_area = self.bg_mask ^ mask_expand  # xor
            self.n_wbg[pad_area] = [0, 0, -1]

        self.n_wbg_vis = (camera_to_object(self.n_wbg) + 1) / 2
        self.n_wbg_vis[~self.bg_mask] = 1

        self.n_wbg_noise = add_noise(self.n_wbg, self.bg_mask)

    def construct_mesh(self):
        self.facets = construct_facets_from_depth_map_mask(self.mask)
        self.surf = pv.PolyData(self.vertices, self.facets)


# H = 200
# H = 64
H = 128
# H = 256
ox = H / 2 - 0.5
oy = H / 2 - 0.5
f = 600
d = 10
# r = 1.6
# r = 0.5
r = 1
# r = 2


K = np.array([[f, 0, ox],
              [0, f, oy],
              [0, 0, 1]], dtype=np.float)
sphere = Data()
sphere.K = np.array([[f, 0, ox],
              [0, f, oy],
              [0, 0, 1]], dtype=np.float)
sphere.n, sphere.mask, sphere.depth_gt, sphere.vertices = perspective_sphere_normal_and_depth(H, K, d=d, r=r)
sphere.construct_mesh()
sphere.curl, *_ = curl_of_normal_map(sphere.n, sphere.mask)
sphere.projection = "perspective"
sphere.fname = "sphere_per"
sphere.n_vis = (camera_to_object(sphere.n)+1)/2
sphere.bg_only_mask = np.isnan(sphere.depth_gt)

# plt.imshow(sphere.n_vis)
# # plt.show()

sphere.n[~sphere.mask] = 0
sphere.n_vis[~sphere.mask] = 1
# pad_mask = np.logical_or(np.isnan(sphere.depth_gt), sphere.depth_gt > (10 - r ** 2 / 10))
sphere.depth_gt[sphere.bg_only_mask] = 10 - r ** 2 / 10
#
sphere.add_noise()
sphere.add_outlier()
sphere.add_outlier_on_noise_map()
sphere.add_background()
v3d = np.ones((H, H, 3))
v3d[sphere.mask] = sphere.vertices





