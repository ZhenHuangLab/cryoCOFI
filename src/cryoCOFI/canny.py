import numpy as np
from scipy import ndimage

def canny(lp_filtered_img, canny_kernel, threshold=0.4):
    # Apply Gaussian smoothing
    smoothed = ndimage.gaussian_filter(lp_filtered_img, sigma=canny_kernel)
    
    # Calculate gradients using Sobel
    dx = ndimage.sobel(smoothed, axis=0)
    dy = ndimage.sobel(smoothed, axis=1)
    
    # Calculate gradient magnitude
    gradient_magnitude = np.hypot(dx, dy)
    
    # Normalize gradient magnitude
    gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
    
    # Threshold
    lp_filtered_img_edge = (gradient_magnitude > threshold).astype(np.float32)
    
    # Clean up small objects
    lp_filtered_img_edge = ndimage.binary_opening(lp_filtered_img_edge)
    
    # Edge thinning using binary erosion with cross structure
    cross = np.array([[0,1,0],
                        [1,1,1],
                        [0,1,0]], dtype=bool)
    lp_filtered_img_edge = ndimage.binary_erosion(lp_filtered_img_edge, 
                                                structure=cross,
                                                iterations=20)
    return lp_filtered_img_edge