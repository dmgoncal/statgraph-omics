import json
import os
import re
from collections import Counter
import numpy as np
import pandas as pd
from . import utils
from . import constants


pd.options.mode.chained_assignment = None

RS = 7

# LOAD/SAVE FROM/TO FILES
def save_df(path: str, df: pd.DataFrame) -> None:
    df.to_csv(path)

def load_df(path: str):
    df = pd.read_csv(path, index_col=0)
    return df

def confirm_create_dir(path: str):
    if not os.path.isdir(path):
        os.mkdir(path)


class LayerDataset:

    @property
    def project_dir(self) -> str:
        return f"{self.data_dir}/{self.project}"

    @property
    def layer_dir(self) -> str:
        return f"{self.project_dir}/{self.layer}"

    @property
    def layer_file(self) -> str:
        return f"{self.layer_dir}/{self.layer}.csv"

    @property
    def counts_file(self) -> str:
        return f"{self.layer_dir}/counts.csv"

    @property
    def rpm_file(self) -> str:
        return f"{self.layer_dir}/rpm.csv"

    @property
    def tpm_file(self) -> str:
        return f"{self.layer_dir}/tpm.csv"

    @property
    def fpkm_file(self) -> str:
        return f"{self.layer_dir}/fpkm.csv"
    
    @property
    def rppa_file(self) -> str:
        return f"{self.layer_dir}/rppa.csv"
    
    @property
    def clinical_file(self) -> str:
        return f"{self.data_dir}/{self.project}/preprocessed_clinical.csv"
    
    @property
    def data_type_codes(self) -> dict:
        return {'rpm': self.rpm_file, 'tpm': self.tpm_file, 'fpkm': self.fpkm_file, 
                'rppa': self.rppa_file, 'counts': self.counts_file}

    
    def drop_target_file(self, target: str) -> str:
        return f"{self.data_dir}/{self.project}/targets/drop_{target}.json"
    
    def replace_target_file(self, target: str) -> str:
        return f"{self.data_dir}/{self.project}/targets/replace_{target}.json"
    

    default_datatype_per_layer = {'mirna': 'rpm', 'mrna': 'tpm', 'protein': 'rppa'}

    @property
    def layer(self):
        self.check_layer_set()
        return self._layer

    @layer.setter
    def layer(self, l: str):
        if l not in LayerDataset.default_datatype_per_layer.keys() and l != None:
            raise ValueError(f'Layer "{l}" not valid. Must be on of the following: {list(LayerDataset.default_datatype_per_layer.keys())}')
        self._layer = l


    def __init__(self, data_dir: str, project: str, layer: str = None):
        self.data_dir = os.path.expanduser(data_dir)
        self.project = project
        self._layer = layer
        # create directories if necessary
        if not os.path.isdir(self.data_dir):                              
            print(f"Data directory '{self.data_dir}' doesn't exist.\nCreating...")
            os.makedirs(self.data_dir)
        if not os.path.isdir(self.project_dir):
            print(f"Project directory '{self.project_dir}' doesn't exist.\nCreating...")
            os.mkdir(self.project_dir)
        if layer is not None and not os.path.isdir(self.layer_dir):
            print(f"Layer directory '{self.layer_dir}' doesn't exist.\nCreating...")
            os.mkdir(self.layer_dir)
    
    def check_layer_set(self):
        if self._layer is None:
            raise ValueError("Layer needs to be set for this operation.")
        return 1


    # CLINICAL DATA RELATED METHODS
    def get_clinical_to_drop(self, target: str) -> list:
        to_drop_file = self.drop_target_file(target)
        try:
            with open(to_drop_file, "rb") as f:
                to_drop = json.load(f)
            return to_drop
        except FileNotFoundError:
            raise FileNotFoundError(f"File {to_drop_file} doesn't exist. Create using: {self.set_clinical_to_drop.__name__}")
    

    def get_clinical_to_replace(self, target: str) -> list:
        to_replace_file = self.replace_target_file(target)
        try:
            with open(to_replace_file, "rb") as f:
                to_drop = json.load(f)
            return to_drop
        except FileNotFoundError:
            raise FileNotFoundError(f"File {to_replace_file} doesn't exist. Create using: {self.set_clinical_to_replace.__name__}")


    def set_clinical_to_drop(self, target: str, values_to_drop: list) -> None:
        to_drop_file = self.drop_target_file(target)
        folder = os.path.dirname(to_drop_file)
        if not os.path.isdir(folder):
            os.mkdir(folder)
        with open(to_drop_file, "w") as f:
            json.dump(values_to_drop, f)
        print(f'Created {to_drop_file}')


    def set_clinical_to_replace(self, target: str, values_to_replace: dict) -> None:
        to_replace_file = self.replace_target_file(target)
        folder = os.path.dirname(to_replace_file)
        if not os.path.isdir(folder):
            os.mkdir(folder)
        with open(to_replace_file, "w") as f:
            json.dump(values_to_replace, f)
        print(f'Created {to_replace_file}')


    def get_clinical_data(self) -> pd.DataFrame:
        """Returns a pandas DataFrame with the clinical data for this project"""

        clinical_file = self.clinical_file
        try:
            df = pd.read_csv(clinical_file, index_col=0)
        except FileNotFoundError:
            print(f"Clinical file '{clinical_file}' doesn't exist.\nCreating...")
            utils.generate_clinical(self.project, clinical_file)
            preprocess_clinical(self.project, self.data_dir)
            df = load_df(clinical_file)
        return df


    # RAW DATA
    def get_raw_data(self, data_type: str) -> pd.DataFrame:
        file = self.data_type_codes[data_type]
        try:
            df = load_df(file)
        except KeyError:
            raise ValueError(f'Data type {data_type} not supported. Must be one of the following: {list(self.data_type_codes.keys())}')
        except FileNotFoundError:
            print(f"Layer file '{file}' doesn't exist.\nCreating...")
            utils.generate_layer(self.project, self.layer, self.data_dir, file, 3, 1)
            preprocess_omics(self.project, self.layer, self.data_dir)
            df = load_df(file)
        return df

    def set_raw_data(self, data_type: str, df: pd.DataFrame) -> None:
        try:
            file = self.data_type_codes[data_type]
        except KeyError:
            raise ValueError(f'Data type {data_type} not supported. Must be one of the following: {list(self.data_type_codes.keys())}')
        save_df(file, df)
    

    def log_transform_data(self, df: pd.DataFrame, data_type: str, target: str) -> pd.DataFrame:
        if data_type=='default':
            data_type = LayerDataset.default_datatype_per_layer[self.layer]

        if target is not None:
            target_col = df[target]
            df = df.drop(columns=[target])

        if data_type == 'rpm':
            total_counts_per_sample = self.get_raw_data("counts").sum(axis="columns")
            df, total_counts_per_sample = utils.match_id_levels(df, total_counts_per_sample)
            total_counts_per_sample = (10 ** 6) / total_counts_per_sample
            for ind, row in df.iterrows():
                df.loc[ind, :] += total_counts_per_sample[ind]
        elif data_type == 'tpm':
            counts_df = self.get_raw_data("counts")
            df, counts_df = utils.match_id_levels(df, counts_df)
            counts_df = counts_df.loc[df.index, df.columns]
            to_add = df / counts_df
            to_add = to_add.fillna(to_add.mean().mean())  # imputation for cases with 0 counts, fill with mean
            df = df + to_add
        else:
            raise ValueError(f"Function 'log_transform_data' not implemented for data type: {data_type}")

        df = np.log2(df)
        if target is not None:
            df[target] = target_col
        return df


    # MULTIOMICS DATA
    def get_multiomics_dge_data(self, target, layers, pvalue, filter_only=False, data_type="default", log_transform=True) -> pd.DataFrame:
        original_layer = self.layer # save current layer to reset at the end
        dge_datasets = []
        for layer in layers:
            self.layer = layer
            if layer == "protein":
                data = self.get_data_with_target(data_type='rppa', target=target)
            else:
                data = self.get_dge_data(target, pvalue, filter_only, data_type, log_transform)
            print(f"{layer} shape: {data.shape} ")
            dge_datasets.append(data)
        self.layer = original_layer

        df_multi = pd.concat(dge_datasets, axis='columns', join='inner')
        del dge_datasets
        df_multi = df_multi.loc[:,~df_multi.columns.duplicated()] # drop duplicated target column
        # more target column to the end
        target_column = df_multi.pop(target)
        df_multi.insert(len(df_multi.columns), target, target_column)
        print(f"multi-omics shape: {df_multi.shape}")
        return df_multi


    # CLINICAL + TARGET DATASETS
    def get_data_with_target(self, data_type: str, target: str, log_transform=False):
        if data_type=='default':
            data_type = LayerDataset.default_datatype_per_layer[self.layer]

        df = self.get_raw_data(data_type)
        clinical_df = self.get_clinical_data()

        # log transform raw data
        if log_transform:
            df = self.log_transform_data(df, data_type, target=None)

        # getting target column from clinical data
        try:
            target_column = clinical_df[target]
        except KeyError:
            raise ValueError(f"Invalid argument {{target}} '{target}'. Must be one of the following: "
                             f"{list(clinical_df.columns)}")
        
        # droping values in target
        try:
            values_to_drop = self.get_clinical_to_drop(target)
            target_column = target_column[~target_column.isin(values_to_drop)]
        except FileNotFoundError:
            print(f"No file with values to drop. Continuing.")
            pass

        # replacing values in target
        try:
            replace_dict = self.get_clinical_to_replace(target)
            target_column = target_column.replace(replace_dict)
        except FileNotFoundError:
            print(f"No file with values to replace. Continuing.")
            pass

        df, target_column = utils.match_id_levels(df, target_column)
        df = pd.merge(df, target_column, left_index=True, right_index=True)

        # define positive (1) and negative (0) classes in binary, 0, .., n otherwise
        target_values = df[target].value_counts(sort=True, ascending=False).keys()
        target_replace = dict(zip(target_values, range(len(target_values))))
        df[target] = df[target].replace(target_replace).infer_objects(copy=False)

        return df


    # Get complete datasets for this project
    def get_layer(self, n_rows: int = None) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.layer_file, nrows=n_rows, index_col=0)
        except FileNotFoundError:
            print(f"Layer file '{self.layer_file}' doesn't exist.\nCreating...")
            self.generate_layer()
            df = pd.read_csv(self.layer_file, nrows=n_rows, index_col=0)
        return df

    def generate_layer(self):
        # allow for the use of a 'test' layer that behaves just like a normal layer
        #   until this point where it gets the data equivalent to the mirna layer
        layer = 'mirna' if self.layer == 'test' else self.layer
        utils.generate_layer(self.project, layer,  self.data_dir, self.layer_file)

    # Get information about columns of layer
    def get_columns_of_layer(self) -> list:
        """Gets the list of columns for this layer"""
        try:
            columns = pd.read_csv(self.layer_file, nrows=0).columns.tolist()
        except FileNotFoundError:
            print(f"Layer file '{self.layer_file}' doesn't exist.\nCreating...")
            self.generate_layer()
            columns = pd.read_csv(self.layer_file, nrows=0).columns.tolist()
        return columns

    def get_types_of_columns(self) -> Counter:
        """Gets types of columns when there are multiple columns for a single patient"""
        cols = self.get_columns_of_layer()
        types = list(map(lambda x: x.split("TCGA")[0], cols))
        return Counter(types)

    def get_columns_of_type(self, type_of_column: str) -> list:
        """Gets the list of columns of a certain type, as returned by 'get_types_of_columns'"""
        columns = self.get_columns_of_layer()
        if type_of_column == "":
            type_of_column = "TCGA"
        r = re.compile(f"{type_of_column}.*")
        columns = list(filter(r.match, columns))
        return columns

    # Get partial datasets for layer
    def get_layer_by_columns(self, columns: list) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.layer_file, usecols=columns, index_col=0)
        except FileNotFoundError:
            print(f"Layer file '{self.layer_file}' doesn't exist.\nCreating...")
            self.generate_layer()
            df = pd.read_csv(self.layer_file, usecols=columns, index_col=0)
        return df

    def get_layer_by_column_type(self, type_of_column: str, keep_unique_columns=True) -> pd.DataFrame:
        """Gets a pandas DataFrame with only a specified type of column, as obtained by 'get_types_of_columns'."""
        columns = self.get_columns_of_type(type_of_column)

        if keep_unique_columns:
            counter_columns = self.get_types_of_columns()
            unique_columns = [item[0] for item in counter_columns.items() if item[1] == 1]
            columns = columns + unique_columns

        df = self.get_layer_by_columns(columns)
        df.columns = [col.replace(type_of_column, "") for col in df.columns]
        return df

    def get_layer_by_sample(self, sample_id: str, keep_unique_columns=True) -> pd.DataFrame:
        columns = self.get_columns_of_layer()
        columns = [col for col in columns if sample_id in col]
        if keep_unique_columns:
            counter_columns = self.get_types_of_columns()
            unique_columns = [item[0] for item in counter_columns.items() if item[1] == 1]
            columns = columns + unique_columns

        df = self.get_layer_by_columns(columns)
        return df


    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'LayerDataset object with project="{self.project}", layer="{self._layer}", and DATA_DIR="{self.data_dir}"'
    

#######################################################################################
# Data Preprocessing Functions ########################################################
#######################################################################################

def preprocess_clinical(proj: str, data_dir: str):
    '''
    Processes and cleans clinical data.
    '''
    dataset = LayerDataset(data_dir, proj, None)
    df = dataset.get_clinical_data()

    # Replacements
    replacements = {
        "race": "not reported",
        "vital_status": "Not Reported",
        "treatments_pharmaceutical_treatment_or_therapy": "not reported",
        "treatments_radiation_treatment_or_therapy": "not reported"
    }
    for col, val in replacements.items():
        df[col] = df[col].replace(val, np.nan)

    # Remove nan values and constant features
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, df.nunique(dropna=False) > 1]
    df = df[~(df["race"].isna())]
    df = df.dropna(axis=0, how="all")

    # Drop non-relevant columns
    df = df.drop(columns=["age_at_diagnosis", "days_to_birth", "year_of_birth"], errors="ignore")
    df = df.drop(columns=["prior_treatment"], errors="ignore")
    df = df.drop(columns=["ethnicity"], errors="ignore")
    df = df.drop(columns=["bcr_patient_barcode", "demographic_id", "diagnosis_id", "exposure_id",
                          "treatments_pharmaceutical_treatment_id", "treatments_radiation_treatment_id"], errors="ignore")
    
    # Exclude no prior + synchronous malignanciy cases
    df = df[~((df["prior_malignancy"] == "no") & (df["synchronous_malignancy"] == "Yes"))]
    if proj == "TCGA-KIRC":
        df = df.drop(columns=["morphology", "synchronous_malignancy"], errors="ignore")
    else:
        df = df.drop(columns=["site_of_resection_or_biopsy", "icd_10_code", "morphology", "synchronous_malignancy"], errors="ignore")

    # Write preprocessed data
    out_path = os.path.join(data_dir, proj, "preprocessed_clinical.csv")
    df.to_csv(out_path, index=False)

    # Processes and cleans the SELECTED target columns.
    for target in constants.TARGETS:
        config = constants.PROJECT_CONFIG.get(proj, {}).get(target, None)
        if config is None:
            return  # Skip if config is not defined for the target
        dataset = LayerDataset(data_dir, proj, layer=None)
        dataset.set_clinical_to_drop(target, config["drop"])
        dataset.set_clinical_to_replace(target, config["replace"])
    

def preprocess_omics(project: str, layer: str, data_dir: str, crossmap_cutoff: int = 50, zero_cutoff: int = 0.95):
    '''
    Processes and cleans omics data, depending on specified layer.
    '''
    if layer=='mirna':
        # miRNA Processing
        dataset = LayerDataset(data_dir, project, "mirna")
        
        # Cross-mapping analysis
        cross_df = dataset.get_layer_by_column_type("cross-mapped_")
        cross_counts = (cross_df == "Y").sum(axis=1)
        to_remove_cross = cross_counts[cross_counts > crossmap_cutoff].index
        
        # Sparsity analysis
        counts_df = dataset.get_layer_by_column_type("read_count_")
        print(f"Original shape ({project},miRNA-counts): {counts_df.shape}")
        zero_fraction = (counts_df == 0).sum(axis=1) / counts_df.shape[1]
        to_remove_zero = zero_fraction[zero_fraction >= zero_cutoff].index
        
        # Process counts data
        filtered_counts = counts_df.drop(index=to_remove_cross, errors="ignore")
        filtered_counts = filtered_counts.drop(index=to_remove_zero, errors="ignore")
        dataset.set_raw_data("counts", filtered_counts.T)

        # Process rpm data
        rpm_df = dataset.get_layer_by_column_type("reads_per_million_miRNA_mapped_")
        print(f"Original shape ({project},miRNA-rpm): {rpm_df.shape}")
        filtered_rpm = rpm_df.drop(index=to_remove_cross, errors="ignore")
        filtered_rpm = filtered_rpm.drop(index=to_remove_zero, errors="ignore")
        dataset.set_raw_data("rpm", filtered_rpm.T)

        print(f"Final shape (samples x miRNA-counts): {filtered_counts.shape}")
        print(f"Final shape (samples x miRNA-rpm): {filtered_rpm.shape}")
    
    elif layer=='mrna':
        # mRNA Processing Functions
        dataset = LayerDataset(data_dir, project, "mrna")
        
        # Process counts data
        df = dataset.get_layer_by_column_type('unstranded_')
        print(f"Original shape ({project},mRNA-counts): {df.shape}")
        df = df.dropna().drop(columns=["gene_name", "gene_type"])
        # Sparsity analysis
        zero_count = ((df == 0).sum(axis="columns") / df.shape[1])
        remove_zero = zero_count >= zero_cutoff
        remove_zero = df[remove_zero].index
        df_counts = df.drop(index=remove_zero).T
        dataset.set_raw_data(data_type="counts", df=df_counts)
        
        # Process TPM data
        df_tpm = dataset.get_layer_by_column_type("tpm_unstranded_")
        print(f"Original shape ({project},mRNA-tpm): {df_tpm.shape}")
        df_tpm = df_tpm.dropna().drop(columns=["gene_name", "gene_type"]).drop(index=remove_zero).T
        dataset.set_raw_data(data_type="tpm", df=df_tpm)
        
        # Process FPKM data
        df_fpkm = dataset.get_layer_by_column_type("fpkm_unstranded_")
        print(f"Original shape ({project},mRNA-fpkm): {df_fpkm.shape}")
        df_fpkm = df_fpkm.dropna().drop(columns=["gene_name", "gene_type"]).drop(index=remove_zero).T
        dataset.set_raw_data(data_type="fpkm", df=df_fpkm)

        print(f"Final shape (samples x mRNA-counts): {df_counts.shape}")
        print(f"Final shape (samples x mRNA-tpm): {df_tpm.shape}")
        print(f"Final shape (samples x mRNA-fpkm): {df_fpkm.shape}")

    elif layer=='protein':
        # Protein Processing Functions
        dataset = LayerDataset(data_dir, project, "protein")
        
        # Get data and drop unnecessary columns
        df = dataset.get_layer_by_column_type('')
        print(f"Original shape ({project},proteins): {df.shape}")
        df = df.drop(columns=["lab_id", "catalog_number", "set_id", "peptide_target"])
        
        # Handle NaN values
        df = df.dropna(axis=0, how="any")
        
        # Process RPPA data
        dataset.set_raw_data(data_type="rppa", df=df.T)
        print(f"Final shape (samples x proteins): {df.shape}")
