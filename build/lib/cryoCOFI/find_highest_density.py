import numpy as np
import ctypes
from cryoCOFI import get_lib_path

def find_highest_density(img, mask):
    if img.ndim != 2 or mask.ndim != 2:
        raise ValueError("Both img and mask must be 2D numpy arrays.")
    
    # Load the CUDA library
    # cuda_lib = ctypes.CDLL(os.path.abspath("./lib/find_highest_density_double.so"))
    cuda_lib = ctypes.CDLL(get_lib_path('find_highest_density_double'))
    # Define the function prototype
    cuda_lib.find_highest_density_cuda.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
        ctypes.c_int
    ]
    cuda_lib.find_highest_density_cuda.restype = ctypes.c_double

    # Ensure the arrays are contiguous and have the correct data types
    img_flat = np.ascontiguousarray(img.flatten(), dtype=np.float64)
    mask_flat = np.ascontiguousarray(mask.flatten(), dtype=np.int32)
    
    # Call the CUDA function
    diff = cuda_lib.find_highest_density_cuda(img_flat, mask_flat, img_flat.size)
    
    return diff

# def find_highest_density(img, mask):
#     cuda_lib = ctypes.CDLL(os.path.abspath("./lib/find_highest_density.so"))

#     # Define the function prototype
#     cuda_lib.find_highest_density_cuda.argtypes = [
#         np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
#         np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
#         ctypes.c_int
#     ]
#     cuda_lib.find_highest_density_cuda.restype = ctypes.c_float

#     if img.ndim != 2 or mask.ndim != 2:
#         raise ValueError("Both img and mask must be 2D numpy arrays.")
    
#     # Ensure the arrays are contiguous and have the correct data types
#     img_flat = np.ascontiguousarray(img.flatten(), dtype=np.float32)
#     mask_flat = np.ascontiguousarray(mask.flatten(), dtype=np.int32)
    
#     # Call the CUDA function
#     diff = cuda_lib.find_highest_density_cuda(img_flat, mask_flat, img_flat.size)
#     return diff