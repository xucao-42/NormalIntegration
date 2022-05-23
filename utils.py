import numpy as np
from sklearn.preprocessing import normalize
import cv2
from scipy.sparse import coo_matrix
from PIL import Image, ImageChops
import os

def move_left(mask):
    return np.pad(mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]


def move_right(mask):
    return np.pad(mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]


def move_top(mask):
    return np.pad(mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]


def move_bottom(mask):
    return np.pad(mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]


def move_top_left(mask):
    return np.pad(mask, ((0, 1), (0, 1)), "constant", constant_values=0)[1:, 1:]


def move_top_right(mask):
    return np.pad(mask, ((0, 1), (1, 0)), "constant", constant_values=0)[1:, :-1]


def move_bottom_left(mask):
    return np.pad(mask, ((1, 0), (0, 1)), "constant", constant_values=0)[:-1, 1:]


def move_bottom_right(mask):
    return np.pad(mask, ((1, 0), (1, 0)), "constant", constant_values=0)[:-1, :-1]


def normalize_normal_map(N):
    H, W, C = N.shape 
    N = np.reshape(N, (-1, C))
    N = normalize(N, axis=1)
    N = np.reshape(N, (H, W, C))
    return N

def file_path(string):
    if os.path.isfile(string) or os.path.isdir(string):
        return string
    else:
        raise FileNotFoundError(string)


def construct_facets_from_depth_map_mask(mask):
    idx = np.zeros_like(mask, dtype=np.int)
    idx[mask] = np.arange(np.sum(mask))

    facet_move_top_mask = move_top(mask)
    facet_move_left_mask = move_left(mask)
    facet_move_top_left_mask = move_top_left(mask)

    facet_top_left_mask = np.logical_and.reduce((facet_move_top_mask, facet_move_left_mask, facet_move_top_left_mask, mask))
    facet_top_right_mask = move_right(facet_top_left_mask)
    facet_bottom_left_mask = move_bottom(facet_top_left_mask)
    facet_bottom_right_mask = move_bottom_right(facet_top_left_mask)

    return np.hstack((4 * np.ones((np.sum(facet_top_left_mask), 1)),
               idx[facet_top_left_mask][:, None],
               idx[facet_bottom_left_mask][:, None],
               idx[facet_bottom_right_mask][:, None],
               idx[facet_top_right_mask][:, None])).astype(np.int)


def construct_vertices_from_depth_map_and_mask(mask, depth_map, step_size=1):
    H, W = mask.shape
    yy, xx = np.meshgrid(range(W), range(H))
    xx = np.flip(xx, 0)
    xx = xx * step_size
    yy = yy * step_size

    vertices = np.zeros((H, W, 3))
    vertices[..., 0] = xx
    vertices[..., 1] = yy
    vertices[..., 2] = depth_map
    return vertices[mask]


def map_depth_map_to_point_clouds(depth_map, mask, K):
    # x
    # |  z
    # | /
    # |/
    # o ---y
    H, W = mask.shape
    yy, xx = np.meshgrid(range(W), range(H))
    xx = np.flip(xx, axis=0)
    u = np.zeros((H, W, 3))
    u[..., 0] = xx
    u[..., 1] = yy
    u[..., 2] = 1
    u = u[mask].T  # 3 x m
    p_tilde = (np.linalg.inv(K) @ u).T  # m x 3
    return p_tilde * depth_map[mask, np.newaxis]


def apply_jet_on_single_error_map(err_map):
    mu = np.nanmean(err_map)
    sigma = np.nanstd(err_map)
    err_map = err_map / (mu + 3 * sigma)
    err_map[err_map > 1] = 1
    err_map[np.isnan(err_map)] = 1
    err_jet = cv2.applyColorMap((255 * err_map).astype(np.uint8), cv2.COLORMAP_JET)
    return err_jet


def apply_jet_on_multiple_error_maps(err_maps, sigma_multiplier=3):
    mu = np.nanmean(np.array(err_maps))
    sigma = np.nanstd(np.array(err_maps))
    err_jets = []
    for err_map in err_maps:
        err_map = err_map / (mu + sigma_multiplier * sigma)
        err_map[err_map > 1] = 1
        nan_mask = np.isnan(err_map)
        err_map[nan_mask] = 1
        err_jet = cv2.applyColorMap((255 * err_map).astype(np.uint8), cv2.COLORMAP_JET)
        err_jet[nan_mask] = 255
        err_jets.append(err_jet)
    return err_jets


def camera_to_object(n):
    no = n.copy()
    no[..., 2] = -no[..., 2]
    temp0 = no[..., 0].copy()
    temp1 = no[..., 1].copy()
    no[..., 1] = temp0
    no[..., 0] = temp1
    return no


def boundary_excluded_mask(mask):
    top_mask = np.pad(mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]
    bottom_mask = np.pad(mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]
    left_mask = np.pad(mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]
    right_mask = np.pad(mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]
    be_mask = np.logical_and.reduce((top_mask, bottom_mask, left_mask, right_mask, mask))

    # discard single point
    top_mask = np.pad(be_mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]
    bottom_mask = np.pad(be_mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]
    left_mask = np.pad(be_mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]
    right_mask = np.pad(be_mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]
    bes_mask = np.logical_or.reduce((top_mask, bottom_mask, left_mask, right_mask))
    be_mask = np.logical_and(be_mask, bes_mask)
    return be_mask


def boundary_expansion_mask(mask):
    left_mask = np.pad(mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]
    right_mask = np.pad(mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]
    top_mask = np.pad(mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]
    bottom_mask = np.pad(mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]

    be_mask = np.logical_or.reduce((left_mask, right_mask, top_mask, bottom_mask))
    return be_mask


def curl_of_normal_map(n, mask):
    nx = n[..., 0]
    ny = n[..., 1]
    nz = n[..., 2]

    zx = -nx / nz
    zy = -ny / nz

    top_mask = np.pad(mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]
    bottom_mask = np.pad(mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]
    left_mask = np.pad(mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]
    right_mask = np.pad(mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]

    top_mask = np.logical_and(top_mask, mask)
    bottom_mask = np.logical_and(bottom_mask, mask)
    left_mask = np.logical_and(left_mask, mask)
    right_mask = np.logical_and(right_mask, mask)

    num_top = np.sum(top_mask)
    num_bottom = np.sum(bottom_mask)
    num_left = np.sum(left_mask)
    num_right = np.sum(right_mask)
    num_mask = np.sum(mask)

    idx_array = np.ones_like(mask, dtype=np.int) * np.nan
    idx_array[mask] = np.arange(np.sum(mask))

    right_row_idx = np.tile(np.arange(num_right), 2)
    right_column_idx = np.concatenate((idx_array[right_mask].flatten(),
                                       idx_array[left_mask].flatten())).astype(np.int)
    right_data = np.array([-1] * num_right + [1] * num_right)
    right_convolution = coo_matrix((right_data, (right_row_idx, right_column_idx)),
                                   shape=(num_right, num_mask))

    right_forward_flatten = right_convolution @ zx[mask].flatten()
    right_forward = np.ones_like(mask) * np.nan
    right_forward[right_mask] = right_forward_flatten
    #

    left_row_idx = np.tile(np.arange(num_left), 2)
    left_column_idx = np.concatenate((idx_array[left_mask].flatten(),
                                       idx_array[right_mask].flatten())).astype(np.int)
    left_data = np.array([1] * num_left + [-1] * num_left)
    left_convolution = coo_matrix((left_data, (left_row_idx, left_column_idx)),
                                   shape=(num_left, num_mask))

    left_forward_flatten = left_convolution @ zx[mask].flatten()
    left_forward = np.ones_like(mask) * np.nan
    left_forward[left_mask] = left_forward_flatten

    top_row_idx = np.tile(np.arange(num_top), 2)
    top_column_idx = np.concatenate((idx_array[top_mask].flatten(),
                                       idx_array[bottom_mask].flatten())).astype(np.int)
    top_data = np.array([-1] * num_top + [1] * num_bottom)
    top_convolution = coo_matrix((top_data, (top_row_idx, top_column_idx)),
                                   shape=(num_top, num_mask))

    top_forward_flatten = top_convolution @ zy[mask].flatten()
    top_forward = np.ones_like(mask) * np.nan
    top_forward[top_mask] = top_forward_flatten

    bottom_row_idx = np.tile(np.arange(num_bottom), 2)
    bottom_column_idx = np.concatenate((idx_array[bottom_mask].flatten(),
                                       idx_array[top_mask].flatten())).astype(np.int)
    bottom_data = np.array([1] * num_top + [-1] * num_bottom)
    bottom_convolution = coo_matrix((bottom_data, (bottom_row_idx, bottom_column_idx)),
                                   shape=(num_bottom, num_mask))

    bottom_forward_flatten = bottom_convolution @ zy[mask].flatten()
    bottom_forward = np.ones_like(mask) * np.nan
    bottom_forward[bottom_mask] = bottom_forward_flatten

    z_xy = np.nanmean(np.concatenate((right_forward[..., None],
                                      left_forward[..., None]), -1), -1)

    z_yx = np.nanmean(np.concatenate((top_forward[..., None],
                                      bottom_forward[..., None]), -1), -1)



    # z_xy_cv2 = cv2.filter2D(zx, -1, kernel=np.array([[0, 0, 0],
    #                                             [-0.5, 0, 0.5],
    #                                              [0, 0, 0]]))
    # z_yx_cv2 = cv2.filter2D(zy, -1, kernel=np.array([[0, 0.5, 0],
    #                                             [0, 0, 0],
    #                                              [0, -0.5, 0]]))

    curl = np.abs(z_xy - z_yx)
    # curl_cv2 = np.abs(z_xy_cv2 - z_yx_cv2)
    return curl, z_yx, z_xy, zx, zy


def crop_a_set_of_images(*image_path):
    from PIL import ImageChops, Image
    imgs = []
    bboxes = []
    for im_path in image_path:
        im = Image.open(im_path)
        bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -5)
        bbox = diff.getbbox()

        imgs.append(im)
        bboxes.append(bbox)
    bbox_aggre = np.asarray(bboxes)
    bbox_min = np.min(bbox_aggre, 0)
    bbox_max = np.max(bbox_aggre, 0)
    bbox_common = (bbox_min[0], bbox_min[1], bbox_max[2], bbox_max[3])
    for idx, img in enumerate(imgs):
        img = img.crop(bbox_common)
        img.save(image_path[idx])
    pass


def angular_error_map(N1, N2):
    dot = np.sum(np.multiply(N1, N2), axis=-1)
    dot = np.clip(dot, -1., 1.)
    return np.rad2deg(np.arccos(dot))


def crop_mask(mask):
    if mask.dtype is not np.uint8:
        mask = mask.astype(np.uint8) * 255
    im = Image.fromarray(mask)
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, 0)
    bbox = diff.getbbox()
    return bbox


def crop_image_by_mask(img, mask):
    bbox = crop_mask(mask)
    return img.copy()[bbox[1]:bbox[3], bbox[0]:bbox[2]]


def mkdir(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
