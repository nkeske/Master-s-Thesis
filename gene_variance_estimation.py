import anndata
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from scipy.stats import ttest_ind
import scanpy as sc
import numpy as np


def variable_gene_selection(expression_file, train_sample_file):

    adata = sc.read_h5ad(expression_file)
    train_sample_names = np.loadtxt(train_sample_file, dtype=str).tolist()

    mask = adata.obs['Donor ID'].isin(train_sample_names)
    training_adata = adata[mask]

    sc.pp.highly_variable_genes(training_adata, n_top_genes=2000, flavor="seurat_v3", n_bins=20)

    sc.pl.highly_variable_genes(training_adata)

    highly_variable_genes_names = training_adata.var_names[training_adata.var['highly_variable']]
    highly_variable_genes_names.to_series().to_csv('highly_variable_genes_train.csv', index=False)

    adata_highly_var = training_adata[:, training_adata.var['highly_variable']]


    columns_to_keep = ['Donor ID', 'Subclass', 'Cognitive Status', 'Overall AD neuropathological Change']
    adata_highly_var.obs = adata_highly_var.obs.loc[:, columns_to_keep]
    print('done5')

    adata_highly_var.write('Variance_Estimation_2000_full.h5ad')
    return adata_highly_var

expression_file = sc.read_h5ad("SEAAD_MTG_RNAseq.h5ad")
train_sample_file = 'train_sample_names_40.txt'

variable_gene_selection(expression_file, train_sample_file)