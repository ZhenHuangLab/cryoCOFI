import cupy as cp
import mrcfile

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

def flag_masking(img, a, b, radius, map_cropping=20, dist_thr_inside_edge=20, map_cropping_flag=2, edge_flag=3, flag_for_dist_thr_inside_edge=4):
    '''
    Generate a mask for calculating the average density inside and outside the circle.
    Args:
        img: image
        a: center x
        b: center y
        radius: radius
        map_cropping: size of the map cropping (pixels)
        dist_thr_inside_edge: distance threshold inside edge (pixels). 
                              Extending the edge so that particles within this distance will also be removed.
        map_cropping_flag: flag for the map cropping
        edge_flag: flag for the edge
        flag_for_dist_thr_inside_edge: flag for the distance threshold inside edge
    Returns:
        mask: mask
    '''
    mask = cp.zeros_like(img)
    y, x = cp.ogrid[:img.shape[0], :img.shape[1]]
    distance_map = cp.sqrt((x - b)**2 + (y - a)**2)
    mask[distance_map < radius - 1 ] = 1

    mask[(distance_map > radius - 1 - dist_thr_inside_edge) & (distance_map < radius - 1)] = flag_for_dist_thr_inside_edge

    mask[(distance_map > radius - 1 ) & (distance_map < radius + 1)] = edge_flag

    # cropping the whole image
    mask[0:map_cropping, :] = map_cropping_flag
    mask[img.shape[0]-map_cropping:img.shape[0], :] = map_cropping_flag
    mask[:, 0:map_cropping] = map_cropping_flag
    mask[:, img.shape[1]-map_cropping:img.shape[1]] = map_cropping_flag

    return mask
