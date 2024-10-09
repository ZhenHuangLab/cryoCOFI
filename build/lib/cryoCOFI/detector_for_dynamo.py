from .carbon_film_detector import detector_for_mrc
import tqdm
import numpy as np
import os
import pandas as pd

def read_dynamo_doc(doc_path):
    # 获取一个dict，key为mrc_path_index，value为mrc_path
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
    # Define column dtypes
    dtypes = {i: int for i in range(40)}  # Set all columns to int by default
    float_columns = [6, 7, 8, 23, 24, 25]  # Columns to be set as float (0-based index)
    for col in float_columns:
        dtypes[col] = float

    # Read the file with specified dtypes
    df = pd.read_csv(tbl_path, sep=' ', header=None, dtype=dtypes)
    return df

def save_dynamo_tbl(df, out_path):
    # Convert int columns to int (in case they were changed to float during processing)
    int_columns = [col for col in df.columns if col not in [6, 7, 8, 23, 24, 25]]
    df[int_columns] = df[int_columns].astype(int)

    # Save the DataFrame without index and header
    df.to_csv(out_path, sep=' ', header=None, index=None, float_format='%.5f')

def read_dynamo_tbl_tomogram_index(df):
    # 读取.tbl文件，得到表格，返回第19列的值，代表tomogram的index
    # 去除重复的tomogram_index
    tomogram_indices = df[19].unique()
    return tomogram_indices

def read_dynamo_tbl_particle_list(df, tomogram_index):
    # 读取tomogram_indices列表，搜索到tomogram_index相同值的行，返回序号
    # 读取.tbl文件，得到表格，根据上述行序号，返回那一段table slice

    # 找到与tomogram_index相同的行
    mask = df[19] == tomogram_index
    # 根据mask，得到行序号
    row_indices = np.where(mask)[0]
    # 根据行序号，得到table slice
    df_slice = df.iloc[row_indices]
    
    return df_slice

def multi_mrc_processing_dynamo(doc_path,
                        tbl_path,
                        out_path,
                        low_pass,
                        kernel_radius,
                        sigma_color,
                        sigma_space,
                        diameter,
                        edge,
                        mode_threshold,
                        edge_quotient_threshold,
                        verbose):

    mrc_index_paths = read_dynamo_doc(doc_path)

    df = read_dynamo_tbl(tbl_path)

    # Create df_modified with the same dtypes as df
    df_modified = pd.DataFrame(columns=df.columns).astype(df.dtypes)

    tomogram_indices = read_dynamo_tbl_tomogram_index(df)

    for tomogram_index in tqdm.tqdm(tomogram_indices, desc="Processing tomograms", position=0, dynamic_ncols=True, unit="tg"):
        df_slice = read_dynamo_tbl_particle_list(df, tomogram_index)

        mrc_path = mrc_index_paths[tomogram_index]

        if not os.path.exists(mrc_path):
            print(f"Warning: Skip tomogram {tomogram_index} {mrc_path} because not exist.")
            df_modified = pd.concat([df_modified, df_slice], ignore_index=True)
            continue

        mask = detector_for_mrc(mrc_path,
                                low_pass,
                                kernel_radius,
                                sigma_color,
                                sigma_space,
                                diameter,
                                edge,
                                mode_threshold,
                                edge_quotient_threshold,
                                show_fig=False,
                                verbose=verbose)
        if mask is False:
            # 如果没有碳膜，不会动原本的particle，直接将df_slice添加到df_modified
            df_modified = pd.concat([df_modified, df_slice], ignore_index=True)
            if verbose:
                print(f"Skip tomogram {tomogram_index} {mrc_path} because no carbon film detected.")
        else:
            if verbose:
                print(f"Processing tomogram {tomogram_index} {mrc_path} with carbon film detected.")
            # 有碳膜时，需要根据path的序号实现另一个循环，判断tomogram中的xy坐标是在圆弧内还是圆弧外，在圆弧外则从列表中删去该坐标，在圆弧内则保留
            for _, row in tqdm.tqdm(df_slice.iterrows(), desc="Screening particles", position=1, dynamic_ncols=True, unit="ptcl", leave=False):
                x = row[23]
                y = row[24]
                if mask[y, x] == 1:
                    df_modified = pd.concat([df_modified, pd.DataFrame(row).T], ignore_index=True)

    save_dynamo_tbl(df_modified, out_path)
    print(f"New tbl file saved to {out_path}.")