import scanpy as sc
import pandas as pd
import numpy as np

expression_file = "Variance_Estimation_2000_full.h5ad"
train_sample_file = "train_sample_names.txt"
test_sample_file = "test_sample_names.txt"
validation_set1_file = "validation_sample_names1.txt"

validation_set2_file = "validation_sample_names2.txt"


def data_loading_processing(expression_file, train_sample_file, test_sample_file, validation_set1_file,
                            validation_set2_file,
                            target_column_name):
    
    # target_column_name will be eighter 'Cognitive Status' or 'Neuropathological Change'

    # Load the filtered anndata file
    adata_filtered = sc.read_h5ad(expression_file)
    replacement_dict = {'Low': 'AD', 'Intermediate': 'AD', 'High': 'AD'}

    # Read the train, test and validation sample names from the files
    train_sample_names = np.loadtxt(train_sample_file, dtype=str).tolist()
    test_sample_names = np.loadtxt(test_sample_file, dtype=str).tolist()
    validation_sample_names1 = np.loadtxt(validation_set1_file, dtype=str).tolist()
    validation_sample_names2 = np.loadtxt(validation_set2_file, dtype=str).tolist()

    # Generate an dataframe contains gene expressin data from anndata object
    expression_matrix = pd.DataFrame(adata_filtered.X.A, columns=adata_filtered.var_names,
                                     index=adata_filtered.obs_names)

    # Add metadata columns
    expression_matrix['Donor ID'] = adata_filtered.obs['Donor ID']
    expression_matrix[target_column_name] = adata_filtered.obs[target_column_name]
    expression_matrix['Subclass'] = adata_filtered.obs['Subclass']
    if target_column_name == 'Overall AD neuropathological Change':
        expression_matrix[target_column_name] = expression_matrix[target_column_name].replace(replacement_dict)

    train_sample_names = sorted(train_sample_names)
    test_sample_names = sorted(test_sample_names)
    validation_sample_names1 = sorted(validation_sample_names1)
    validation_sample_names2 = sorted(validation_sample_names2)

    return expression_matrix, train_sample_names, test_sample_names, validation_sample_names1, validation_sample_names2

