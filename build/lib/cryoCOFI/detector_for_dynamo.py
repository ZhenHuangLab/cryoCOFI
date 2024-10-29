from .carbon_film_detector import detector_for_mrc
import tqdm
import numpy as np
import os
import pandas as pd

def read_dynamo_doc(doc_path):
    '''
    Read the mrc file paths from the Dynamo doc file.
    Args:
        doc_path: path to the Dynamo doc file
    Returns:
        mrc_index_paths: dict, key is the tomogram index, value is the mrc file path
    '''
    mrc_index_paths = {}
    with open(doc_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) > 1:
                mrc_path_index = int(parts[0])
                mrc_path = ' '.join(parts[1:])
                mrc_index_paths[mrc_path_index] = mrc_path
    return mrc_index_paths

def read_dynamo_tbl(tbl_path):
    '''
    Read the Dynamo tbl file.
    Args:
        tbl_path: path to the Dynamo tbl file
    Returns:
        df: DataFrame
    '''
    # Read the file and let pandas infer the dtypes
    df = pd.read_csv(tbl_path, sep=' ', header=None)
    return df

def save_dynamo_tbl(df, out_path):
    '''
    Save the Dynamo tbl file.
    Args:
        df: DataFrame
        out_path: path to the output file (tbl file)
    '''
    # Convert int columns to int (in case they were changed to float during processing)
    int_columns = [col for col in df.columns if col not in [6, 7, 8, 23, 24, 25]]
    df[int_columns] = df[int_columns].astype(int)

    # Save the DataFrame without index and header
    df.to_csv(out_path, sep=' ', header=None, index=None, float_format='%.5f')

def read_dynamo_tbl_tomogram_index(df):
    '''
    Read the tomogram indices from the Dynamo tbl file.
    Args:
        df: DataFrame
    Returns:
        tomogram_indices: list, tomogram indices
    '''
    # remove the duplicate tomogram indices
    tomogram_indices = df[19].unique()
    return tomogram_indices

def read_dynamo_tbl_particle_list(df, tomogram_index):
    '''
    Read the particle list from the Dynamo tbl file.
    Args:
        df: DataFrame
        tomogram_index: tomogram index
    Returns:
        df_slice: DataFrame, particle list
    '''
    # find the rows with the same tomogram index
    mask = df[19] == tomogram_index
    # get the row indices
    row_indices = np.where(mask)[0]
    # get the table slice
    df_slice = df.iloc[row_indices]
    
    return df_slice

def multi_mrc_processing_dynamo(doc_path,
                        tbl_path,
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
    Process the mrc files from the Dynamo doc file and tbl file.
    Args:
        doc_path: path to the Dynamo doc file
        tbl_path: path to the Dynamo tbl file
        out_path: path to the output file (tbl file)
        low_pass: low pass filter (angstrom)
        detector_type: type of the detector
        kernel_radius: kernel radius (pixels)
        sigma_color: sigma color
        sigma_space: sigma space
        canny_kernel: canny kernel size
        diameter: diameter (angstrom)
        map_cropping: map cropping (pixels)
        dist_thr_inside_edge: distance inside edge (pixels)
        mode_threshold: mode threshold
        edge_quotient_threshold: edge quotient threshold
        verbose: whether to show the print information
    '''
    mrc_index_paths = read_dynamo_doc(doc_path)

    df = read_dynamo_tbl(tbl_path)

    # Create df_modified with the same dtypes as df
    df_modified = pd.DataFrame(columns=df.columns).astype(df.dtypes)

    tomogram_indices = read_dynamo_tbl_tomogram_index(df)

    # Add inner progress bar
    inner_pbar = tqdm.tqdm([], desc="Screening particles", 
                          position=1, dynamic_ncols=True, unit="ptcl", leave=True)

    try:
        for tomogram_index in tqdm.tqdm(tomogram_indices, desc="Processing tomograms", 
                                       position=0, dynamic_ncols=True, unit="tg", leave=True):
            df_slice = read_dynamo_tbl_particle_list(df, tomogram_index)
            mrc_path = mrc_index_paths[tomogram_index]

            if not os.path.exists(mrc_path):
                print(f"Warning: Skip tomogram {tomogram_index} {mrc_path} because not exist.")
                df_modified = pd.concat([df_modified, df_slice], ignore_index=True)
                continue

            mask = detector_for_mrc(mrc_path,
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

            if mask is False:
                df_modified = pd.concat([df_modified, df_slice], ignore_index=True)
                inner_pbar.reset(total=1)
                inner_pbar.bar_format = "{desc}: NA%|{bar:5} no carbon film detected {bar:5}|NA/NA [{elapsed}<NA, NA ptcls]"
                inner_pbar.update(0)
                if verbose:
                    print(f"Skip tomogram {tomogram_index} {mrc_path} because no carbon film detected.")
            else:
                if verbose:
                    print(f"Processing tomogram {tomogram_index} {mrc_path} with carbon film detected.")
                inner_pbar.bar_format = None
                inner_pbar.reset(total=len(df_slice))
                
                # screening the particles
                for _, row in df_slice.iterrows():
                    inner_pbar.update(1)
                    x = row[23]
                    y = row[24]
                    if mask[int(y), int(x)] == 1:
                        df_modified = pd.concat([df_modified, pd.DataFrame(row).T], ignore_index=True)

    except KeyboardInterrupt:
        inner_pbar.bar_format = None
        inner_pbar.close()
        raise
    finally:
        inner_pbar.close()

    # save the modified tbl file
    save_dynamo_tbl(df_modified, out_path)
    print(f"New tbl file saved to {out_path}.")
