import numpy as np
import os
import pandas as pd
import tqdm
import starfile

# TODO: add readrelion module
def read_relion_star(star_path):
    '''
    Read the Relion star file.
    Args:
        star_path: path to the Relion star file
    Returns:
        df: DataFrame, 
    '''

    df = starfile.read(star_path, always_dict=True)
    return df
    