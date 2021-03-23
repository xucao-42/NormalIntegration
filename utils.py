import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import cv2
from scipy.sparse import coo_matrix
import time
import pyvista as pv
from PIL import Image, ImageChops

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
    """
    N is a unnormalized normal map of shape H_W_3. Normalize N across the third dimension.
    :param N:
    :return:
    """
    H, W, C = N.shape 
    N = np.reshape(N, (-1, C))
    N = normalize(N, axis=1)
    N = np.reshape(N, (H, W, C))
    return N


def construct_facet_for_depth(mask):
    idx = np.zeros_like(mask, dtype=np.int)
    idx[mask] = np.arange(np.sum(mask))

    facet_move_top_mask = np.pad(mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]
    facet_move_left_mask = np.pad(mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]
    facet_move_top_left_mask = np.pad(mask, ((0, 1), (0, 1)), "constant", constant_values=0)[1:, 1:]

    facet_top_left_mask = np.logical_and.reduce((facet_move_top_mask, facet_move_left_mask, facet_move_top_left_mask, mask))
    facet_top_right_mask = np.pad(facet_top_left_mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]
    facet_bottom_left_mask = np.pad(facet_top_left_mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]
    facet_bottom_right_mask = np.pad(facet_top_left_mask, ((1, 0), (1, 0)), "constant", constant_values=0)[:-1, :-1]


    return np.hstack((4 * np.ones((np.sum(facet_top_left_mask), 1)),
               idx[facet_top_left_mask][:, None],
               idx[facet_bottom_left_mask][:, None],
               idx[facet_bottom_right_mask][:, None],
               idx[facet_top_right_mask][:, None])).astype(np.int)


def apply_jet_on_single_error_map(err_map):
    mu = np.nanmean(err_map)
    sigma = np.nanstd(err_map)
    err_map = err_map / (mu + 3 * sigma)
    err_map[err_map > 1] = 1
    nan_mask = np.isnan(err_map)
    err_map[nan_mask] = 1
    err_jet = cv2.applyColorMap((255 * err_map).astype(np.uint8), cv2.COLORMAP_JET)
    err_jet[nan_mask] = 255
    return err_jet


def apply_jet_on_multiple_error_maps(err_maps):
    mu = np.nanmean(np.array(err_maps))
    sigma = np.nanstd(np.array(err_maps))
    err_jets = []
    for err_map in err_maps:
        err_map = err_map / (mu + 3 * sigma)
        err_map[err_map > 1] = 1
        nan_mask = np.isnan(err_map)
        err_map[nan_mask] = 1
        err_jet = cv2.applyColorMap((255 * err_map).astype(np.uint8), cv2.COLORMAP_JET)
        err_jet[nan_mask] = 255
        err_jets.append(err_jet)
    return err_jets

def mesh_plot(vertices_list, facet_list, filepath=None):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    X = vertices_list[..., 0]
    Y = vertices_list[..., 1]
    Z = vertices_list[..., 2]
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.plot_trisurf(X, Y, Z, triangles=facet_list, linewidth=0.2, antialiased=True, shade=True)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if filepath:
        plt.savefig(filepath)
    else:
        plt.show()

def world_to_object(n):
    no = n.copy()
    no[..., 2] = -no[..., 2]
    temp0 = no[..., 0].copy()
    temp1 = no[..., 1].copy()
    no[..., 1] = temp0
    no[..., 0] = temp1
    return no

def write_off_from_indexed_face_set(vertices, faces, file_path, num_vertex_per_facet=3):
    """
    Refer to Polygon Mesh Processing. pp.22.
    :param vertice: a Nx3 ndarray or list
    :param faces: a Nx3 ndarray or list
    :param file_path:
    :return:
    """
    if type(vertices) is not list:
        vertices = list(vertices)
    if type(faces) is not list:
        faces = list(faces)
    with open(file_path, "w") as f:
        f.write("OFF" + '\n')
        f.write("{0} {1} {2}\n".format(len(vertices), len(faces), 0))
        for vertice in vertices:
            f.write("{0:.8f} {1:.8f} {2:.8f}\n".format(vertice[0], vertice[1], vertice[2]))
        for face in faces:
            if num_vertex_per_facet == 4:
                f.write("4 {0} {1} {2} {3}\n".format(face[0], face[1], face[2], face[3]))
            elif num_vertex_per_facet == 3:
                f.write("3 {0} {1} {2}\n".format(face[0], face[1], face[2]))
        f.flush()

def hide_all_plot(img, colorbar=True, fname=None, title="", vmin=0, vmax=10, cmap="viridis"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([1], [1])
    ax.set_xticks([])
    ax.set_yticks([])
    plot = plt.imshow(img, vmax=vmax, vmin=vmin, cmap=cmap)
    if colorbar:
        cbar = plt.colorbar(plot, format="%.0e")
        cbar.ax.tick_params(labelsize=18)

    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    ax.axis('off')
    ax.tick_params(axis=u'both', which=u'both', length=0)
    plt.title(title)
    if fname:
        plt.savefig(fname)
    else:
        plt.show()
    plt.clf()
    plt.close()


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

def generate_dis_mesh(d, K=None):
    r = (d - 1) / 2
    facet_idx = np.arange(d * d).reshape((d, d)) + 1  # facet idx begin from 1

    top_left_mask = np.pad(facet_idx, ((0, 1), (0, 1)), "constant", constant_values=0) != 0
    top_right_mask = np.pad(facet_idx, ((0, 1), (1, 0)), "constant", constant_values=0) != 0
    bottom_left_mask = np.pad(facet_idx, ((1, 0), (0, 1)), "constant", constant_values=0) != 0
    bottom_right_mask = np.pad(facet_idx, ((1, 0), (1, 0)), "constant", constant_values=0) != 0

    vertex_mask = np.logical_or.reduce((top_right_mask, top_left_mask, bottom_right_mask, bottom_left_mask))
    vertex_idx = np.zeros((d + 1, d + 1), dtype=np.int)
    vertex_idx[vertex_mask] = np.arange(np.sum(vertex_mask)) + 1  # vertex idx begin from 1

    num_vertex = np.max(vertex_idx)
    num_facet = np.max(facet_idx)

    yy, xx = np.meshgrid(range(d + 1), range(d + 1))
    xx = np.max(xx) - xx
    xx = xx.astype(np.float)
    yy = yy.astype(np.float)
    # xx -= 0.5
    # yy -= 0.5
    v_0 = np.zeros((d + 1, d + 1, 3))
    v_0[..., 0] = xx
    v_0[..., 1] = yy

    if K is not None:
        K_1 = np.linalg.inv(K)
        # create homogenouse coordinate
        v_0[..., 2] = 1
        v_0_f = v_0.reshape(-1, 3).T
        view_direction = (K_1 @ v_0_f).T.reshape((d + 1, d + 1, 3))

        v = np.zeros((d + 1, d + 1, 3))

        alpha_top_left = 1 / (view_direction[..., 0] + view_direction[..., 2])
        alpha_top_right = 1 / (view_direction[..., 2] - view_direction[..., 0])
        v[:int(r) + 2, :int(r) + 1, :] = (alpha_top_left[..., None] * view_direction)[:int(r) + 2, :int(r) + 1, :]
        v[:int(r) + 2, int(r) + 1:, :] = (alpha_top_right[..., None] * view_direction)[:int(r) + 2, int(r) + 1:, :]
        v[int(r) + 2:, ...] = view_direction[int(r) + 2:, ...] / view_direction[int(r) + 2:, :, 2:3]

        scale = np.sqrt(np.sum(v[..., 2] ** 2))
        v /= scale
    else:
        dis = 100
        slop = np.sqrt(2) * (xx - r)
        v_0[..., 0] -= r
        v_0[..., 1] -= r
        v_0[:int(r) + 2, :int(r) + 1, 2] = dis - slop[:int(r) + 2, :int(r) + 1]
        v_0[:int(r) + 2, int(r) + 1:, 2] = dis + slop[:int(r) + 2, :int(r) + 1]
        v_0[int(r) + 2:, :, 2] = dis
        v = v_0.copy()



    vertices_list = v[vertex_mask].reshape((num_vertex, 3))
    face_list_all = []
    for i in range(d):
        for j in range(d):
            if j != int(r) or i>int(r):
                face_list_all.append([vertex_idx[i, j], vertex_idx[i + 1, j], vertex_idx[i, j + 1]])

    for i in range(d):
        for j in range(d):
            if j != int(r) or i>int(r):
                face_list_all.append([vertex_idx[i, j + 1], vertex_idx[i + 1, j], vertex_idx[i + 1, j + 1]])

    facet_list = np.array([i for i in face_list_all if 0 not in i])
    facet_list -= 1  # the vertex index in a mesh starts from 0
    H, W = facet_list.shape
    facet_list_surf = np.hstack((np.ones((H, 1)) * 3, facet_list)).astype(np.int)
    surf = pv.PolyData(vertices_list, facet_list_surf)
    return surf, v


def save_normal_map(n, fname):
    n = world_to_object(n)
    n = (n + 1) / 2
    cv2.imwrite(fname, cv2.cvtColor((n * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))


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


if __name__ == "__main__":
    # img_path = ["../selected_results/dragon_gt.png",
# #     #             "../selected_results/dragon_l1.png",
# #     #             "../selected_results/dragon_l2.png"]
# #     # crop_a_set_of_images(*img_path)

    surf, _ = generate_dis_mesh(65)
    surf.plot()

    # plane
    # H = 34
    # n = generate_dis_normal_map(H)
    # mask = np.ones((H, H), dtype=np.bool)


    # sphere
    # n, mask, _ = generate_normal_map_and_depth(32)

    # mitsuba rendered normal
    # n = np.load("../../data/bunny_normal_map/orthographic/orthographic.npy")
    # n = np.load("../data/bunny_normal_map/perspective/perspective.npy")
    # n = np.load("../data/armadillo/perspective.npy")
    # n = np.load("../../data/dragon/perspective.npy")
    # n = np.load("../data/budda/perspective.npy")
    # n = np.load("../data/dragon2/perspective.npy")

    # normal_norm = np.sqrt(np.sum(n ** 2, axis=-1, keepdims=1))
    # n /= normal_norm
    # normal_norm = np.sqrt(np.sum(n ** 2, axis=-1, keepdims=1))
    # mask = np.squeeze(np.isclose(normal_norm, 1, rtol=1e-1))
    # mask = boundary_excluded_mask(mask)
    # H = 64
    # ox = H / 2 - 0.5
    # oy = H / 2 - 0.5
    # f = 600
    # K = np.array([[f, 0, ox],
    #               [0, f, oy],
    #               [0, 0, 1]], dtype=np.float)
    # n, t, v0_3d, tv, vv = perspective_sphere_normal_and_depth(H, K, d=10, r=0.5)
    # from mesh_from_normal import MeshFromNormal
    # mesh = MeshFromNormal(n, K, "o")
    # mesh.save("sphere_orth.off")
    # curl, *_ = curl_of_normal_map(n, mask) + 1e-15
    # plt.imshow(curl)
    # plt.show()
    # hide_all_plot(curl, vmax=1, colorbar=True)
