import os
import pandas as pd
import numpy as np
import sys
import time
import subprocess
from typing import List
import glob
import re


MODULE_DIR = os.path.dirname(__file__)
R_SCRIPT_DIR = os.path.join(MODULE_DIR, 'r_dir/')
TEMP_ID_FILE = os.path.join(MODULE_DIR, 'temp_ids_timestamp.RData')

# Utils ###############################################################################
# Utility functions ###################################################################
#######################################################################################

def execute(cmd):
    """Creates subprocess to run received command and yields the output as it is produced"""

    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in iter(popen.stdout.readline, ""):
        yield line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def get_id_column(df: pd.DataFrame) -> str:
    """Finds the ID column of a DataFrame by searching for the column with unique values for all rows"""
    
    nrows = df.shape[0]
    for column in df.columns:
        if len(df[column].value_counts()) == nrows:
            return column
    
    # if reached, then no column with IDs only
    # select column with most unique values
    nu = df.nunique().idxmax()
    return nu


def reduce_numeric_64_to_32(df: pd.DataFrame, verbose = 0) -> pd.DataFrame:
    """Reduces int64 and float64 columns to int32 and float32 if possible"""

    max_int32 = np.iinfo(np.int32).max
    max_float32 = np.finfo(np.float32).max
    counter_int = 0
    counter_float = 0
    for col in df.columns:
        coltype = df[col].dtype
        if coltype == np.int64:
            if (df[col].abs() < max_int32).all():
                df[col] = df[col].astype("int32")
                counter_int +=1
        elif coltype == np.float64:
            if (df[col].abs() < max_float32).all():
                df[col] = df[col].astype("float32")
                counter_float+=1
                
    if verbose >= 1: print(f"Reduced {counter_int} ints and {counter_float} floats")
    return df


# Generate Functions ##################################################################
# Used to run R scripts that download the data from TCGA and store it in disk #########
#######################################################################################

def generate_ids(project: str, ids_file_path: str, verbose: int = 1) -> None:
    """Generates the ids.RData file for the corresponding project and stores it in disk"""

    print(["Rscript", f'{R_SCRIPT_DIR}get_ids.R', project, ids_file_path])
    for output in execute(["Rscript", f'{R_SCRIPT_DIR}get_ids.R', project, ids_file_path]):
        if verbose >= 1: print(output, end="")


def generate_clinical(project: str, clinical_file_path: str, verbose: int = 1) -> None:
    """Generates a processed dataframe with clinical data from the data of TCGA and stores it in disk"""

    for output in execute(["Rscript", f'{R_SCRIPT_DIR}get_clinical.R', project, clinical_file_path]):
        if verbose >= 1: print(output, end="")


def generate_layer(project: str, layer: str, data_dir: str, layer_file: str, divider: int = 1,
                   verbose: int = 1) -> None:
    """Generates a processed dataframe (or multiple) from the data of TCGA and stores it in disk"""
    temp_file = TEMP_ID_FILE.replace('timestamp', f'{layer}_{str(int(time.time()))}')
    try:
        generate_ids(project, temp_file, verbose=verbose)  # create temp file with layer IDs for get_omics.R
        for output in execute(
                ["Rscript", f'{R_SCRIPT_DIR}get_omics.R', project, layer, "TRUE", str(divider), data_dir, layer_file, temp_file]):
            if verbose >= 1: print(output, end="")
    finally:
        os.remove(temp_file)  # delete temp file with layer IDs
    merge_segments_of_layer(layer, layer_file, verbose=verbose)  # Merges the segments and stores complete layer in disk



def match_id_levels(df1: pd.DataFrame, df2: pd.DataFrame, deal_with_duplicates="delete") -> List[pd.DataFrame]:
    """Matches ID levels of two dataframes by reducing the longest type of ID to match the shortest"""
    valid_deal_with_duplicates = ["delete"]

    df_list = [df1, df2]
    id1 = df_list[0].index[0].split("-")
    id2 = df_list[1].index[0].split("-")

    len_list = [len(id1), len(id2)]
    min_index = np.argmin(len_list)
    reduce_index = 1 - min_index

    new_index = ["-".join(i.split("-")[:len_list[min_index]]) for i in df_list[reduce_index].index]

    df_list[reduce_index].index = new_index

    if deal_with_duplicates == "delete":
        df_list[reduce_index] = df_list[reduce_index][~df_list[reduce_index].index.duplicated(keep='first')]
    else:
        raise ValueError(
            f"Invalid deal_with_duplicates argument '{deal_with_duplicates}'. Should be one of the following: {valid_deal_with_duplicates}.")

    return df_list


def merge_segments_of_layer(layer: str, layer_file: str, verbose=0) -> None:
    """Merges the segments of the layer dataset as produced by 'generate_layer' into a complete layer dataset"""
    layer = layer.lower()
    mb_size = 1024 ** 2

    search_string = layer_file.replace('.csv', '_*.csv')
    print(search_string)
    segments_files = sorted(glob.glob(search_string))
    if len(segments_files) == 0:
        raise FileNotFoundError(f"No files of type '{search_string}' found.")

    final = pd.read_csv(segments_files[0])
    final = reduce_numeric_64_to_32(final, verbose=1)

    id_col = get_id_column(final)

    for i, file in enumerate(segments_files[1:]):
        if verbose >= 1:
            print(
                f"Segment {i + 2} out of {len(segments_files)}\t\tCurrent size of dataframe: {round(sys.getsizeof(final) / mb_size)} MB",
                end='\r')
        t = pd.read_csv(file)
        t = reduce_numeric_64_to_32(t, verbose=1)

        diff_cols = t.columns.difference(final.columns).tolist() + [id_col]
        final = final.merge(t.loc[:, diff_cols], how="inner", on=[id_col])
        del t

    # fix column names of cnv data
    if layer == "cnv":
        r = re.compile(f"TCGA.*")
        columns = list(filter(r.match, final.columns))
        new_columns = ["_".join(c.split('_')[1:]) + '_' + c.split(',')[0] for c in columns]
        replace_dict = dict(zip(columns, new_columns))
        final.rename(columns=replace_dict, inplace=True)

    final.to_csv(layer_file, index=False)
    if verbose >= 1:
        print(f'File: "{layer_file}" saved successfully')
