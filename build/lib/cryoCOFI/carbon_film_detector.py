import cupy as cp
import matplotlib.pyplot as plt
import mrcfile
from .low_pass_filter import low_pass_filter_gaussian
from .hough_transform import hough_transform_for_radius
from .bicanny import *
from .average_z import average_along_z
from ._utils import read_pixel_size, flag_masking
from .find_highest_density import find_highest_density

def detector_for_mrc(tg_path,
            low_pass,
            kernel_radius,
            sigma_color,
            sigma_space,
            diameter,
            edge,
            mode_threshold,
            edge_quotient_threshold,
            show_fig,
            verbose):
    
    with mrcfile.open(tg_path) as mrc:
        data = mrc.data
    pixel_size = read_pixel_size(tg_path)
    data = cp.array(data)

    if data.ndim == 3:
        average = average_along_z(tg_path)
    else:
        average = data

    lp_filtered_img = low_pass_filter_gaussian(average, low_pass, pixel_size)

    lp_bi_img = bilateral_filter(lp_filtered_img, kernel_radius, sigma_color, sigma_space)
    lp_filtered_img_edge = edge_detector(lp_bi_img)
    radius = int((diameter / 2) / pixel_size)
    center_x, center_y, hough_img = hough_transform_for_radius(lp_filtered_img_edge, radius)
    a, b = center_y-radius, center_x-radius
    
    # generate the mask
    mask = flag_masking(lp_filtered_img, a, b, radius, edge)

    mode_diff = find_highest_density(lp_filtered_img, mask.get())

    if show_fig:
        show_figure(hough_img, lp_filtered_img_edge, lp_filtered_img, mask, center_x, center_y, radius)
    
    if verbose:
        print(f"Carbon film edge arc: center_x {center_x}, center_y {center_y}")

    if mode_diff < mode_threshold:
        if verbose:
            print(f"mode_diff: {mode_diff} < {mode_threshold} \n Carbon film not detected!")
        return False
    else:
        # calculate edge_quotient
        edge_map_mean = np.mean(lp_filtered_img_edge)
        masked_edge_map_mean = np.mean(lp_filtered_img_edge[mask.get() == 3])
        if verbose:
            print(f"lp_filtered_img_edge mean: {edge_map_mean}")
            print(f"masked lp_filtered_img_edge mean: {masked_edge_map_mean}")
        edge_quotient = masked_edge_map_mean / edge_map_mean

        if edge_quotient > edge_quotient_threshold:
            if verbose:
                print(f"edge_quotient: {edge_quotient} > {edge_quotient_threshold} \n Carbon film detected!")
            return mask
        else:
            if verbose:
                print(f"edge_quotient: {edge_quotient} < {edge_quotient_threshold} \n Carbon film not detected!")
            return False

        

def show_figure(hough_img, lp_filtered_img_edge, lp_filtered_img, mask, center_x, center_y, radius):
    # plt.figure(1)
    # plt.hist(lp_filtered_img[mask.get() == 1], bins=100, alpha=0.5, label='Inside')
    # plt.hist(lp_filtered_img[mask.get() == 0], bins=100, alpha=0.5, label='Outside')
    # plt.xlabel('Density')
    # plt.ylabel('Frequency')
    # plt.legend()

    a,b = center_y-radius, center_x-radius
    theta = np.linspace(0, 2 * np.pi, num=360, dtype=np.float32)
    x = np.round(a + radius * np.cos(theta))
    y = np.round(b + radius * np.sin(theta))
    valid_indices = (x >= 0) & (x < lp_filtered_img.shape[0]) & (y >= 0) & (y < lp_filtered_img.shape[1])


    # plt.figure(2)
    plt.subplot(141)
    plt.imshow(hough_img, cmap='gray')
    plt.plot(center_x, center_y, 'ro')
    plt.title('Hough Space')

    plt.subplot(142)
    plt.imshow(lp_filtered_img_edge, cmap='gray')
    plt.plot(y[valid_indices], x[valid_indices], 'r-')
    plt.title('Edge Map')
    plt.subplot(143)
    plt.imshow(lp_filtered_img, cmap='gray')
    plt.plot(y[valid_indices], x[valid_indices], 'r-')
    plt.title('Detection Result')

    plt.subplot(144)
    plt.imshow(mask.get(), cmap='gray')
    plt.title('Mask')
    
    plt.show()
    plt.tight_layout()

if __name__ == '__main__':
    detector_for_mrc()