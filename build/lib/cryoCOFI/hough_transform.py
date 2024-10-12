import numpy as np
import ctypes
from cryoCOFI import get_lib_path

def hough_transform_for_radius(edge_image, radius):
    '''
    Hough transform for the fixed radius.
    Args:
        edge_image: edge image
        radius: radius (pixels)
    Returns:
        best_a: center x
        best_b: center y
        accumulator: accumulator
    '''
    edge_image = edge_image.astype(np.uint8)
    rows, cols = edge_image.shape

    lib = ctypes.CDLL(get_lib_path('weighted_hough_transform'))

    lib.hough_transform_for_radius.argtypes = [
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.hough_transform_for_radius.restype = None

    edge_image_ptr = edge_image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    best_a = ctypes.c_int()
    best_b = ctypes.c_int()

    hough_max_x = 2 * radius + cols
    hough_max_y = 2 * radius + rows
    accumulator_size = hough_max_x * hough_max_y

    accumulator = np.zeros((hough_max_y, hough_max_x), dtype=np.int32)
    accumulator_ptr = accumulator.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    lib.hough_transform_for_radius(
        edge_image_ptr,
        ctypes.c_int(rows),
        ctypes.c_int(cols),
        ctypes.c_int(radius),
        ctypes.byref(best_a),
        ctypes.byref(best_b),
        accumulator_ptr
    )

    return best_a.value, best_b.value, accumulator
