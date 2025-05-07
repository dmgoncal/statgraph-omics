import pandas as pd
from typing import List
import os
import subprocess
import time


MODULE_DIR = os.path.dirname(__file__)
R_SCRIPT_DIR = os.path.join(MODULE_DIR, 'r_dir/')


def execute(cmd):
    """ Creates subprocess to run received command and yields the output as it is produced.
    """
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def deg_filtering(counts_df: pd.DataFrame, target: str, pvalue = None, filter_only = False, n_genes = None, verbose=1) -> List[str]:
    """ Receives a dataframe with counts, executes DEG analysis, and returns the selected genes as list of str.
        (Does not return the target variable)
    """
    if pvalue is None and n_genes is None and not filter_only:
        raise ValueError("If 'filter_only' is False then 'pvalue' or 'n_genes' must be set.")
    
    if pvalue is not None and filter_only:
        raise Warning(f"Parameter 'pvalue = {pvalue}' will be ignored due to 'filter_only = True'.")
    
    if n_genes is not None and filter_only:
        raise Warning(f"Parameter 'n_genes = {n_genes}' will be ignored due to 'filter_only = True'.")
    
    if pvalue is not None and n_genes is not None:
        raise Warning(f"Both 'pvalue' and 'n_genes' were set. 'pvalue' will be used and n_genes ignored")
    
    input_file = os.path.join(MODULE_DIR, f"input_deg_filtering_{time.time_ns()}.csv")
    output_file = os.path.join(MODULE_DIR, f"output_deg_filtering_{time.time_ns()}.csv")
    counts_df.to_csv(input_file)

    try:
        for output in execute(["Rscript", f'{R_SCRIPT_DIR}get_deg.R', input_file, output_file, target]):
            if verbose >= 1: print(output, end="")
        degs = pd.read_csv(output_file, index_col=0)
    finally:
        os.remove(input_file)
        os.remove(output_file)

    if filter_only:
        pvalue = 1
        filtered_degs = degs[degs['PValue']<=pvalue]
        genes = filtered_degs.index.tolist()
    elif pvalue is not None:
        filtered_degs = degs[degs['PValue']<=pvalue]
        genes = filtered_degs.index.tolist()
    else:
        filtered_degs = degs.sort_values(by=["PValue"]).iloc[:n_genes, :]
        genes = filtered_degs.index.tolist()
    return genes
