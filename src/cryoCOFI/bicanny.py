import cupy as cp
import numpy as np
import mrcfile
import argparse
import matplotlib.pyplot as plt
from .average_z import average_along_z
from cryoCOFI import get_lib_path
from .low_pass_filter import low_pass_filter_gaussian
import ctypes

def read_pixel_size(mrc_path):
    '''
    Read the pixel size from the mrc file.
    Args:
        mrc_path: path to the mrc file
    Returns:
        pixel_size: pixel size
    '''
    with mrcfile.open(mrc_path) as mrc:
        pixel_size = mrc.voxel_size.x
    return pixel_size

def bilateral_filter(input_image, kernel_radius, sigma_color, sigma_space):
    '''
    Bilateral filter for the image.
    Args:
        input_image: input image
        kernel_radius: kernel radius
        sigma_color: sigma color
        sigma_space: sigma space
    Returns:
        output_image: output image
    '''
    height, width = input_image.shape
    input_image = input_image.astype(np.float32)

    output_image = np.zeros_like(input_image)

    # lib = ctypes.CDLL('./lib/bilateral_filter.so')
    lib = ctypes.CDLL(get_lib_path('bilateral_filter'))
    lib.bilateral_filter.argtypes = [
        ctypes.POINTER(ctypes.c_float),  
        ctypes.POINTER(ctypes.c_float), 
        ctypes.c_int, 
        ctypes.c_int, 
        ctypes.c_int, 
        ctypes.c_float,  
        ctypes.c_float  
    ]
    lib.bilateral_filter.restype = None 

    input_ptr = input_image.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_image.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib.bilateral_filter(
        input_ptr,
        output_ptr,
        ctypes.c_int(width),
        ctypes.c_int(height),
        ctypes.c_int(kernel_radius),
        ctypes.c_float(sigma_color),
        ctypes.c_float(sigma_space)
    )

    return output_image

def gaussian_filter(input_image, kernel_size):
    '''
    Gaussian filter for the image.
    Args:
        input_image: input image
        kernel_size: kernel size
    Returns:
        output_image: output image
    '''
    input_image = input_image.astype(np.float32)
    height, width = input_image.shape

    output_image = np.zeros_like(input_image)

    # lib = ctypes.CDLL('./lib/gaussian_filter.so')
    lib = ctypes.CDLL(get_lib_path('gaussian_filter'))

    lib.gaussian_filter.argtypes = [
        ctypes.POINTER(ctypes.c_float),  
        ctypes.POINTER(ctypes.c_float), 
        ctypes.c_int, 
        ctypes.c_int, 
        ctypes.c_int
    ]
    lib.gaussian_filter.restype = None  

    input_ptr = input_image.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_image.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib.gaussian_filter(
        input_ptr,
        output_ptr,
        ctypes.c_int(width),
        ctypes.c_int(height),
        ctypes.c_int(kernel_size)
    )

    return output_image

def edge_detector(image):
    '''
    Edge detector for the image.
    Args:
        image: input image
    Returns:
        edge_map: edge map
    '''
    image = image.astype(np.float32)
    height, width = image.shape

    edge_map = np.zeros((height, width), dtype=np.uint8)

    # lib = ctypes.CDLL('./lib/edge_detector.so')
    lib = ctypes.CDLL(get_lib_path('edge_detector'))

    lib.edge_detector.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int,
        ctypes.c_int
    ]
    lib.edge_detector.restype = None

    input_ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = edge_map.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    lib.edge_detector(input_ptr, output_ptr, ctypes.c_int(width), ctypes.c_int(height))

    return edge_map


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--lowpass', '-lp', type=int, default=200)

    args = parser.parse_args()
    tg_path = args.input
    low_pass = args.lowpass

    with mrcfile.open(tg_path) as mrc:
        data = mrc.data
    pixel_size = read_pixel_size(tg_path)
    data = cp.array(data)
    average = average_along_z(tg_path)

    lp_filtered_img = low_pass_filter_gaussian(average, low_pass, pixel_size)

    lp_bi_img = bilateral_filter(lp_filtered_img.get(), kernel_radius=5, sigma_color=10.0, sigma_space=10.0)

    lp_g_img = gaussian_filter(lp_filtered_img.get(), kernel_size=5)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(average.get(), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    # plt.imshow(lp_filtered_img.get(), cmap='gray')
    # plt.title('Low-pass Filtered Image')
    # plt.axis('off')

    # plt.subplot(1, 3, 3)
    plt.imshow(lp_bi_img, cmap='gray')
    plt.title('Bilateral Filtered Image')
    plt.axis('off')

    # plt.subplot(1, 3, 3)
    # plt.imshow(lp_g_img, cmap='gray')
    # plt.title('Gaussian Filtered Image')
    # plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(edge_detector(lp_bi_img), cmap='gray')
    plt.title('Edge Image')
    plt.axis('off')
    plt.show()

