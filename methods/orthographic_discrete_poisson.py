from scipy.sparse.linalg import lsqr
from utils import *
import pyvista as pv
import time

def generate_dx_dy(mask, step_size=1):
    # pixel coordinates
    # ^ vertical positive
    # |
    # |
    # |
    # o ---> horizontal positive
    all_depth_idx = np.zeros_like(mask, dtype=np.int)
    all_depth_idx[mask] = np.arange(np.sum(mask))
    num_depth = np.sum(mask)

    num_neighbour_map = np.sum(np.concatenate([move_left(mask)[..., None],
                                               move_right(mask)[..., None],
                                               move_top(mask)[..., None],
                                               move_bottom(mask)[..., None]], -1), axis=-1)
    num_neighbour_map[~mask] = 0

    has_left_mask = np.logical_and(move_right(mask), mask)
    has_right_mask = np.logical_and(move_left(mask), mask)
    has_bottom_mask = np.logical_and(move_top(mask), mask)
    has_top_mask = np.logical_and(move_bottom(mask), mask)

    has_left_mask_left = np.pad(has_left_mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]
    has_right_mask_right = np.pad(has_right_mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]
    has_bottom_mask_bottom = np.pad(has_bottom_mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]
    has_top_mask_top = np.pad(has_top_mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]

    num_has_left = np.sum(has_left_mask)
    num_has_right = np.sum(has_right_mask)
    num_has_top = np.sum(has_top_mask)
    num_has_bottom = np.sum(has_bottom_mask)

    data_term = [-1] * num_has_left + [1] * num_has_left
    row_idx = all_depth_idx[has_left_mask]
    row_idx = np.tile(row_idx, 2)
    col_idx = np.concatenate((all_depth_idx[has_left_mask_left], all_depth_idx[has_left_mask]))
    d_horizontal_neg = coo_matrix((data_term, (row_idx, col_idx)), shape=(num_depth, num_depth))

    data_term = [-1] * num_has_right + [1] * num_has_right
    row_idx = all_depth_idx[has_right_mask]
    row_idx = np.tile(row_idx, 2)
    col_idx = np.concatenate((all_depth_idx[has_right_mask], all_depth_idx[has_right_mask_right]))
    d_horizontal_pos = coo_matrix((data_term, (row_idx, col_idx)), shape=(num_depth, num_depth))

    data_term = [-1] * num_has_top + [1] * num_has_top
    row_idx = all_depth_idx[has_top_mask]
    row_idx = np.tile(row_idx, 2)
    col_idx = np.concatenate((all_depth_idx[has_top_mask], all_depth_idx[has_top_mask_top]))
    d_vertical_pos = coo_matrix((data_term, (row_idx, col_idx)), shape=(num_depth, num_depth))

    data_term = [-1] * num_has_bottom + [1] * num_has_bottom
    row_idx = all_depth_idx[has_bottom_mask]
    row_idx = np.tile(row_idx, 2)
    col_idx = np.concatenate((all_depth_idx[has_bottom_mask_bottom], all_depth_idx[has_bottom_mask]))
    d_vertical_neg = coo_matrix((data_term, (row_idx, col_idx)), shape=(num_depth, num_depth))

    return d_horizontal_pos / step_size, d_horizontal_neg / step_size, d_vertical_pos / step_size, d_vertical_neg / step_size


class OrthographicPoisson:
    # camera coordinates
    # x
    # |  z
    # | /
    # |/
    # o ---y
    # pixel coordinates
    # u
    # |
    # |
    # |
    # o ---v
    def __init__(self, data):
        self.method_name = "orthographic_poisson"
        method_start = time.time()

        # Eq. (4) in "Normal Integration: A Survey."
        p = - data.n[data.mask, 0] / data.n[data.mask, 2]
        q = - data.n[data.mask, 1] / data.n[data.mask, 2]

        # Eqs. (23) and (24) in "Variational Methods for Normal Integration."
        # w/o depth prior
        dvp, dvn, dup, dun = generate_dx_dy(data.mask, data.step_size)
        A = 0.5 * (dup.T @ dup + dun.T @ dun + dvp.T @ dvp + dvn.T @ dvn)
        b = 0.5 * (dup.T + dun.T) @ p + 0.5 * (dvp.T + dvn.T) @ q

        # There should be faster solvers.
        solver_start = time.time()
        z = lsqr(A, b, atol=1e-17, btol=1e-17)[0]
        solver_end = time.time()

        self.solver_runtime = solver_end - solver_start
        self.residual = A @ z - b

        method_end = time.time()
        self.total_runtime = method_end - method_start

        self.depth_map = np.ones_like(data.mask, dtype=np.float) * np.nan
        self.depth_map[data.mask] = z

        # create a mesh model from the depth map
        self.facets = construct_facets_from_depth_map_mask(data.mask)
        self.vertices = construct_vertices_from_depth_map_and_mask(data.mask, self.depth_map, data.step_size)
        self.surface = pv.PolyData(self.vertices, self.facets)

if __name__ == "__main__":
    import pyexr
    import matplotlib.pyplot as plt
    from utils import camera_to_object

    shape = pyexr.open(os.path.join("..", "data", "relief.exr"))

    flash_only_r = shape.get("R")
    flash_only_g = shape.get("G")
    flash_only_b = shape.get("B")
    normal = normalize_normal_map(np.concatenate((flash_only_r, flash_only_g, flash_only_b), -1))
    mask = boundary_excluded_mask(boundary_excluded_mask(np.isclose(np.sum(normal ** 2, -1), 1)))

    # plt.imshow((normal+1)/2)
    # plt.show()
    #
    # plt.imshow(mask)
    # plt.show()


    class Data():
        pass


    obj = Data()
    obj.n = camera_to_object(normal)
    obj.mask = mask
    obj.fname = "relief"
    obj.step_size = 1

    result = OrthographicPoisson(obj)
    result.surface.save("relief.ply", binary=False)






