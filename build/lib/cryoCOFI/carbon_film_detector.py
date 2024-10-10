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
            map_cropping,
            dist_thr_inside_edge,
            mode_threshold,
            edge_quotient_threshold,
            show_fig,
            verbose):
    
    '''
    Carbon film detector for single mrc file.
    Args:
        tg_path: path to the mrc file
        low_pass: low pass filter (angstrom)
        kernel_radius: kernel radius for bilateral filter
        sigma_color: sigma color for bilateral filter
        sigma_space: sigma space for bilateral filter
        diameter: diameter of the carbon film (angstrom)
        map_cropping: size of the map cropping (pixels)
        dist_thr_inside_edge: distance threshold inside edge (pixels)
        mode_threshold: mode threshold. Compare the mode of the density inside and outside the arc detected.
        edge_quotient_threshold: edge quotient threshold. Compare the mean of the edge and the edge map.
        show_fig: whether to show the figure
        verbose: whether to print the information
    Returns:
        mask or False
    '''

    # read mrc file
    with mrcfile.open(tg_path) as mrc:
        data = mrc.data
    pixel_size = read_pixel_size(tg_path)
    data = cp.array(data)

    # average along z
    if data.ndim == 3:
        average = average_along_z(tg_path)
    else:
        average = data

    # low pass filter
    lp_filtered_img = low_pass_filter_gaussian(average, low_pass, pixel_size)

    # bilateral filter
    lp_bi_img = bilateral_filter(lp_filtered_img, kernel_radius, sigma_color, sigma_space)
    # edge detector
    lp_filtered_img_edge = edge_detector(lp_bi_img)

    # calculate the radius
    radius = int((diameter / 2) / pixel_size)

    # hough transform for the fixed radius
    center_x, center_y, hough_img = hough_transform_for_radius(lp_filtered_img_edge, radius)
    a, b = center_y-radius, center_x-radius
    
    # define the flags for the mask to mark the different regions
    map_cropping_flag = 2
    edge_flag = 3
    flag_for_dist_thr_inside_edge = 4

    # generate the mask
    mask = flag_masking(lp_filtered_img, a, b, radius, map_cropping, dist_thr_inside_edge, map_cropping_flag, edge_flag, flag_for_dist_thr_inside_edge)

    # calculate the mode difference
    mode_diff = find_highest_density(lp_filtered_img, mask.get(), inside_flags=(1, flag_for_dist_thr_inside_edge), outside_flags=(0,))

    if show_fig:
        show_figure(tg_path, hough_img, lp_filtered_img_edge, lp_filtered_img, mask, center_x, center_y, radius)
    
    if verbose:
        print(f"Carbon film edge arc: center_x {center_x}, center_y {center_y}")

    if mode_diff < mode_threshold:
        if verbose:
            print(f"mode_diff: {mode_diff} < {mode_threshold} \n Carbon film not detected!")
        return False
    else:
        if verbose:
            print(f"mode_diff: {mode_diff} > {mode_threshold} \n")

        # calculate edge_quotient
        edge_map_mean = np.mean(lp_filtered_img_edge)
        masked_edge_map_mean = np.mean(lp_filtered_img_edge[mask.get() == edge_flag])

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

        

def show_figure(tg_path, hough_img, lp_filtered_img_edge, lp_filtered_img, mask, center_x, center_y, radius):

    '''
    Show the figure for the carbon film detector.
    Args:
        hough_img: hough image
        lp_filtered_img_edge: edge map
        lp_filtered_img: low pass filtered image
        mask: mask
        center_x: center x
        center_y: center y
        radius: radius
    '''
    
    a,b = center_y-radius, center_x-radius
    theta = np.linspace(0, 2 * np.pi, num=360, dtype=np.float32)
    x = np.round(a + radius * np.cos(theta))
    y = np.round(b + radius * np.sin(theta))
    valid_indices = (x >= 0) & (x < lp_filtered_img.shape[0]) & (y >= 0) & (y < lp_filtered_img.shape[1])

    fig, axs = plt.subplots(1, 4, figsize=(20, 5), num=f'readmrc - {tg_path} - cryoCOFI')

    fig.suptitle(f'file: {tg_path}', fontsize=12)
    
    axs[0].imshow(hough_img, cmap='gray')
    axs[0].plot(center_x, center_y, 'ro')
    axs[0].set_title('Hough Space')

    axs[1].imshow(lp_filtered_img_edge, cmap='gray')
    axs[1].plot(y[valid_indices], x[valid_indices], 'r-')
    axs[1].set_title('Edge Map')

    axs[2].imshow(lp_filtered_img, cmap='gray')
    axs[2].plot(y[valid_indices], x[valid_indices], 'r-')
    axs[2].set_title('Detection Result')

    axs[3].imshow(mask.get(), cmap='gray')
    axs[3].set_title('Mask')

    plt.tight_layout()

    # Add caption for the mask image about the different flags
    captions = [
        'Flag 0: Outside Arc (excluded)',
        'Flag 1: Inside Arc (preserved)',
        'Flag 2: Map cropping (excluded)',
        'Flag 3: Edge (excluded)',
        'Flag 4: Distance inside edge (excluded)',
        '@ cryoCOFI'
    ]
    
    fig.text(0.5, 0.03, '; '.join(captions), ha='center', va='center', color='black')
    
    plt.subplots_adjust(bottom=0.1, top=0.9)
    plt.show()

if __name__ == '__main__':
    detector_for_mrc()