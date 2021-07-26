import numpy as np
from scipy.io import loadmat
from utils import camera_to_object, crop_image_by_mask


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

        try:
            data.K = f["K"]
        except:
            pass

    data.n_vis = (camera_to_object(data.n) + 1) / 2
    data.n_vis[~data.mask] = 1
    data.n_vis *= 255
    data.n_vis = crop_image_by_mask(data.n_vis, data.mask)
    data.step_size = 1
    return data

