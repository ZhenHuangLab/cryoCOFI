from .carbon_film_detector import *
import tqdm
import numpy as np
import os
from cryosparc.dataset import Dataset


def multi_mrc_processing_cryosparc(
        cs_path,
        out_path,
        low_pass,
        detector_type,
        kernel_radius,
        sigma_color,
        sigma_space,
        canny_kernel,
        diameter,
        map_cropping,
        dist_thr_inside_edge,
        mode_threshold,
        edge_quotient_threshold,
        verbose):
    '''
    Process the mrc files from the CryoSPARC cs file.
    Args:
        cs_path: path to the CryoSPARC cs file
        out_path: path to the output file (tbl file)
        low_pass: low pass filter (angstrom)
        kernel_radius: kernel radius (pixels)
        sigma_color: sigma color
        sigma_space: sigma space
        canny_kernel: canny detector kernel size
        diameter: diameter (angstrom)
        map_cropping: map cropping (pixels)
        dist_thr_inside_edge: distance inside edge (pixels)
        mode_threshold: mode threshold
        edge_quotient_threshold: edge quotient threshold
        verbose: whether to show the print information
    '''
    particle_dset = Dataset.load(cs_path)
    # Initialize a new empty Dataset
    particle_dset_modified = Dataset.append_many()
    particle_dset_modified = Dataset.innerjoin(particle_dset, particle_dset_modified)
    
    micrograph_paths = np.array(particle_dset['location/micrograph_path'])
    # remove the duplicate micrograph paths
    micrograph_paths = np.unique(micrograph_paths)
    
    # inner progress bar
    inner_pbar = tqdm.tqdm([], desc="Screening particles", 
                          position=1, dynamic_ncols=True, unit="ptcl", leave=True)

    try:
        for micrograph_path in tqdm.tqdm(micrograph_paths, desc="Processing micrographs", 
                                        position=0, dynamic_ncols=True, unit="mg",
                                        leave=True):
            particle_dset_slice = particle_dset.mask(np.array(particle_dset['location/micrograph_path']) == micrograph_path)
            
            if not os.path.exists(micrograph_path):
                print(f"Warning: Skip micrograph {micrograph_path} because not exist.")
                particle_dset_modified.extend(particle_dset_slice)
                continue

            mg_mask = detector_for_mrc(micrograph_path,
                                low_pass,
                                detector_type,
                                kernel_radius,
                                sigma_color,
                                sigma_space,
                                canny_kernel,
                                diameter,
                                map_cropping,
                                dist_thr_inside_edge,
                                mode_threshold,
                                edge_quotient_threshold,
                                show_fig=False,
                                verbose=verbose)

            if mg_mask is False:
                particle_dset_modified.extend(particle_dset_slice)
                inner_pbar.reset(total=1)
                inner_pbar.bar_format = "{desc}: NA%|{bar:5} no carbon film detected {bar:5}|NA/NA [{elapsed}<NA, NA ptcls]"
                inner_pbar.update(0)
                if verbose:
                    print(f"Skip micrograph {micrograph_path} because no carbon film detected.")
            
            else:
                # if carbon film detected, screening the particles
                if verbose:
                    print(f"Processing micrograph {micrograph_path} with carbon film detected.")
                inner_pbar.bar_format = None
                inner_pbar.reset(total=len(particle_dset_slice))
                # screening the particles
                for particle_index in range(len(particle_dset_slice)):
                    inner_pbar.update(1)
                    micrograph_ny = particle_dset_slice['location/micrograph_shape'][particle_index][0]
                    micrograph_nx = particle_dset_slice['location/micrograph_shape'][particle_index][1]
                    location_x = particle_dset_slice['location/center_x_frac'][particle_index] * micrograph_nx
                    location_y = particle_dset_slice['location/center_y_frac'][particle_index] * micrograph_ny
                    if mg_mask[int(location_y), int(location_x)] == 1:
                        particle_dset_modified.extend(particle_dset_slice.slice(start=particle_index, stop=particle_index+1))
    
    except KeyboardInterrupt:
        inner_pbar.bar_format = None
        inner_pbar.close()
        raise
    finally:
        inner_pbar.close()
    
    # save the modified cs file
    particle_dset_modified.save(out_path)
    print(f"New cs file saved to {out_path}.")
                                