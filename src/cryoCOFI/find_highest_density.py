import numpy as np
import ctypes
from cryoCOFI import get_lib_path

def find_highest_density(img, mask, inside_flags=(1, 4), outside_flags=(0,)):
    '''
    Find the density of the highest frequency inside the arc.
    Args:
        img: image
        mask: mask
        inside_flags: flags for the inside
        outside_flags: flags for the outside
    Returns:
        diff: difference
    '''

    if img.ndim != 2 or mask.ndim != 2:
        raise ValueError("Both img and mask must be 2D numpy arrays.")
    
    # Load the CUDA library
    cuda_lib = ctypes.CDLL(get_lib_path('find_highest_density_double'))
    
    # Define the function prototype
    cuda_lib.find_highest_density_cuda.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
        ctypes.c_int
    ]
    cuda_lib.find_highest_density_cuda.restype = ctypes.c_double

    # Ensure the arrays are contiguous and have the correct data types
    img_flat = np.ascontiguousarray(img.flatten(), dtype=np.float64)
    mask_flat = np.ascontiguousarray(mask.flatten(), dtype=np.int32)
    
    # Convert inside_flags and outside_flags to numpy arrays
    inside_flags = np.atleast_1d(inside_flags).astype(np.int32)
    outside_flags = np.atleast_1d(outside_flags).astype(np.int32)
    
    # Call the CUDA function
    diff = cuda_lib.find_highest_density_cuda(
        img_flat, mask_flat, img_flat.size,
        inside_flags, len(inside_flags),
        outside_flags, len(outside_flags)
    )
    
    return diff
