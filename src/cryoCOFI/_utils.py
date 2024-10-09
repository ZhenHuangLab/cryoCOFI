import cupy as cp
import mrcfile

def read_pixel_size(mrc_path):
    with mrcfile.open(mrc_path) as mrc:
        pixel_size = mrc.voxel_size.x
    return pixel_size

def flag_masking(img, a, b, radius, edge=20, edge_flag=2, edge_map_flag=3):
    # generate a mask for calculating the average density inside and outside the circle
    mask = cp.zeros_like(img)
    y, x = cp.ogrid[:img.shape[0], :img.shape[1]]
    distance_map = cp.sqrt((x - b)**2 + (y - a)**2)
    mask[distance_map < radius - 1 ] = 1

    mask[(distance_map > radius - 1) & (distance_map < radius + 1)] = edge_map_flag

    mask[0:edge, :] = edge_flag
    mask[img.shape[0]-edge:img.shape[0], :] = edge_flag
    mask[:, 0:edge] = edge_flag
    mask[:, img.shape[1]-edge:img.shape[1]] = edge_flag

    return mask
