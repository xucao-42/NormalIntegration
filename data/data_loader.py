import os.path

import numpy as np
from scipy.io import loadmat
from utils import camera_to_object, crop_image_by_mask
import cv2

class Data():
    pass


def data_loader(file_path):
    data = Data()
    if file_path.endswith("npy"):
        f = np.load(file_path, allow_pickle=True)

        data.mask = np.squeeze(f.item().get("mask")).astype(bool)
        if data.mask.ndim == 3:
            data.mask = data.mask[..., 0]

        data.n = f.item().get("normal_map")
        data.n = camera_to_object(data.n)

        try:
            data.K = f.item().get("K")
        except:
            pass

    elif file_path.endswith("mat"):
        f = loadmat(file_path)
        data.mask = np.squeeze(f["mask"]).astype(bool)

        if data.mask.ndim == 3:
            data.mask = data.mask[..., 0]

        data.n = f["normal_map"]
        data.n = camera_to_object(data.n)

        try:
            data.K = f["K"]
        except:
            pass

    elif os.path.isdir(file_path):
        normal_map = cv2.cvtColor(cv2.imread(os.path.join(file_path, "normal_map.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_RGB2BGR)
        if normal_map.dtype is np.dtype(np.uint16):
            normal_map = normal_map/65535 * 2 - 1
        else:
            normal_map = normal_map/255 * 2 - 1

        if os.path.exists(os.path.join(file_path, "mask.png")):
            mask = cv2.imread(os.path.join(file_path, "mask.png"), cv2.IMREAD_GRAYSCALE).astype(bool)
        else:
            mask = np.ones(normal_map.shape[:2], bool)

        data.n = camera_to_object(normal_map)
        data.mask = mask

        if os.path.exists(os.path.join(file_path, "K.txt")):
            data.K = np.loadtxt(os.path.join(file_path, "K.txt"))

    data.n_vis = (camera_to_object(data.n) + 1) / 2
    data.n_vis[~data.mask] = 1
    data.n_vis *= 255
    data.n_vis = crop_image_by_mask(data.n_vis, data.mask)
    data.step_size = 1
    return data

