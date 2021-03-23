import numpy as np
from scipy.sparse import coo_matrix, diags, vstack, identity
from scipy.sparse.linalg import eigsh
from utils import construct_facet_for_depth
import pyvista as pv
from scipy.spatial import KDTree

class ZhuSGPerspective:
    # working on camera coordinate
    # x
    # |  z
    # | /
    # |/
    # o ---y
    # image coordinate
    # u
    # |
    # |
    # |
    # o ---v
    def __init__(self, data, setting):
        self.method = "perspective_zhu_and_smith_sg_order_{0}_neighbor_{1}_lambda_smooth_{2}".format(setting.polynomial_order,
                                                                                        setting.num_neighbor,
                                                                                        setting.lambda_smooth)
        mask = data.mask
        num_pixel = np.sum(mask)
        pixel_idx = np.zeros_like(mask, dtype=np.uint)
        pixel_idx[mask] = np.sum(mask)

        # construct neighbor points id matrix
        H, W = mask.shape
        yy, xx = np.meshgrid(range(W), range(H))
        xx = np.max(xx) - xx
        xx = xx.astype(np.float)
        yy = yy.astype(np.float)

        ox = data.K[0, 2]
        oy = data.K[1, 2]
        fx = data.K[0, 0]
        fy = data.K[1, 1]

        u_prime = xx - ox
        v_prime = yy - oy

        # search for nearest neighbourhood pixels for each pixel in image coordinate
        try:
            self.neighbors_idx = data.neighbour_idx
        except:
            img_coordinate = np.concatenate([u_prime[mask][..., None],
                                             v_prime[mask][..., None]], axis=-1)
            query_tree = KDTree(img_coordinate, leafsize=1)
            _, neighbors_idx = query_tree.query(img_coordinate, k=setting.num_neighbor)
            self.neighbors_idx = np.sort(neighbors_idx, axis=-1)

        # retrieve neighbourhood points image coordinate
        center_u = u_prime[mask]
        center_v = v_prime[mask]
        neighbour_u = u_prime[mask][neighbors_idx]
        neighbour_v = v_prime[mask][neighbors_idx]
        poly_list = []
        order_list = []
        for i in range(setting.polynomial_order + 1):
            for j in range(setting.polynomial_order + 1 - i):
                order_list.append((i, j))
                c_i = ((neighbour_u - center_u[..., None]) ** i) * ((neighbour_v - center_v[..., None]) ** j)
                poly_list.append(c_i[..., None])
        C = np.concatenate(poly_list, axis=-1)
        C_pinv = np.linalg.pinv(C)  # num_pixels x num_polynomials x num_neighbour
        a00 = C_pinv[:, order_list.index((0, 0)), :].flatten()  # num_pixels x num_neighbour. smoothness term

        # construct Du and Dv
        Du_data = C_pinv[:, order_list.index((1, 0)), :].flatten()
        Dv_data = C_pinv[:, order_list.index((0, 1)), :].flatten()
        row_id = np.arange(num_pixel)
        row_id = np.repeat(row_id, setting.num_neighbor)
        col_id = neighbors_idx.flatten()

        D_u = coo_matrix((Du_data, (row_id, col_id)), shape=(num_pixel, num_pixel))
        D_v = coo_matrix((Dv_data, (row_id, col_id)), shape=(num_pixel, num_pixel))
        try:
            S = data.S
        except:
            S = coo_matrix((a00, (row_id, col_id)), shape=(num_pixel, num_pixel))
        self.S = S
        # construct the linear system following eq.10 and 11
        U = diags(u_prime[mask])
        V = diags(v_prime[mask])

        if setting.add_noise and setting.add_outlier:
            n1 = data.n_outlier_noise[..., 0][mask]
            n2 = data.n_outlier_noise[..., 1][mask]
            n3 = data.n_outlier_noise[..., 2][mask]
        elif setting.add_noise:
            n1 = data.n_noise[..., 0][mask]
            n2 = data.n_noise[..., 1][mask]
            n3 = data.n_noise[..., 2][mask]
        elif setting.add_outlier:
            n1 = data.n_outlier[..., 0][mask]
            n2 = data.n_outlier[..., 1][mask]
            n3 = data.n_outlier[..., 2][mask]
        else:
            n1 = data.n[..., 0][mask]
            n2 = data.n[..., 1][mask]
            n3 = data.n[..., 2][mask]

        N = vstack([diags(n1),
                    diags(n2),
                    diags(n3)]).T

        Tx = vstack([(U @ D_u + identity(num_pixel)) / fx,
                     V @ D_u / fy,
                     D_u])
        Ty = vstack([U @ D_v / fx,
                     (V @ D_v + identity(num_pixel)) / fy,
                     D_v])

        A = vstack([N @ Tx,
                    N @ Ty,
                    setting.lambda_smooth * (S - identity(num_pixel))])

        _, z = eigsh(A.T @ A, k=1, sigma=0, which="LM")

        if np.sum(z) < 0:
            z = -z
        z_map = np.ones_like(mask, dtype=np.float) * np.nan
        z_map[mask] = np.squeeze(z)
        self.depth = z_map

        # construct a mesh from the depth map
        self.facets = construct_facet_for_depth(mask)
        K_1 = np.linalg.inv(data.K)
        v_0 = np.zeros((H, W, 3))
        v_0[..., 0] = xx
        v_0[..., 1] = yy
        v_0[..., 2] = 1
        v_0_f = v_0[mask].reshape(-1, 3).T
        view_direction = (K_1 @ v_0_f).T
        self.vertices = view_direction * z
        self.surf = pv.PolyData(self.vertices, self.facets)
