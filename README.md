# Nazlıgül Keske  
**Student ID**: 7025902  
**Saarland University**  
**Center for Bioinformatics**  
**Master's Thesis**

---

## Overview

This repository hosts a program applying various machine learning methods—**Decision Trees**, **Random Forests**, **Support Vector Machines** (with different kernel functions), and one neural network approach (**Multi-Layer Perceptrons**)—to multiple data resolutions (**bulk**, **pseudo-bulk**, and **single-cell**). The ultimate goal is to **predict Alzheimer’s disease** from these varying levels of brain data.

---

## Running the Pipeline

1. **Main Notebook**  
   - Run `pipeline_run.ipynb` within this repository.  
   - Ensure that the following files are present in the same directory:
     - `train_sample_names`  
     - `test_sample_names`  
     - `validation_sample_names1`  
     - `validation_sample_names2`  
       > These text files define sample sets for each data split (train/test/validation) as explained in the **Sample-Splitting** section of the thesis.  
     - `SEAAD_MTG_RNAseq.h5ad`  
     - `Variance_Estimation_2000.h5ad`  
       > These files contain **gene expression data**. In particular, `Variance_Estimation_2000.h5ad` is a subset of genes filtered for high variance, produced by `gene_variance_estimation.py`.

2. **Data Inputs & Preprocessing**  
   - `pipeline_run.ipynb` orchestrates loading raw data, creating desired resolution (bulk, pseudo-bulk, single-cell), optimizing ML models, and evaluating predictions against test/validation sets.

---

## Modules & Functions

### `data_loading_processing`
- **Purpose**: Loads the `.h5ad` expression file(s) and reads text files for train/test/validation sample sets.  
- **Inputs**:  
  - `expression_file`  
  - `train_sample_file`  
  - `test_sample_file`  
  - `validation_set1_file`  
  - `validation_set2_file`  
  - `target_column_name` (e.g., `Diagnosis`)  
- **Outputs**:  
  - `expression_matrix` (loaded from `.h5ad`)  
  - `train_sample_names`, `test_sample_names`, `validation_sample_names1`, `validation_sample_names2`

### `data_resolution_prep`
- **Purpose**: Converts the loaded expression matrix into different resolutions.  
- **Methods**:  
  - `bulk_sample_prep()`  
  - `pseudobulk_sample_prep()`  
  - `singlecell_sample_prep()`  
- **Inputs**:  
  - `expression_matrix`, `train_sample_names`, `test_sample_names`, `validation_sample_names1`, `validation_sample_names2`, `target_column_name`  
- **Outputs**:  
  - `training_data`, `training_metadata`  
  - `test_data`, `test_metadata`  
  - `validation_data1`, `validation_metadata1`  
  - `validation_data2`, `validation_metadata2`  
  - `test_sample_labels`

### `gene_variance_estimation`
- **Purpose**: Performs variance analysis on the `SEAAD_MTG_RNAseq.h5ad` dataset to identify highly variable genes.  
- **Inputs**:  
  - `expression_file`  
  - `train_sample_file`  
- **Process**: Generates `Variance_Estimation_2000_full.h5ad`, a pruned file retaining only highly variable genes.

### `optimization`
- **Purpose**: Runs hyperparameter optimization for each model type (Decision Tree, Random Forest, SVMs, and MLP).  
- **Inputs**:  
  - `X_train`, `y_train`  
  - `X_val`, `y_val`  
  - `classifier_type` (e.g., `"svm-rbf"`, `"random_forest"`, `"mlp"`)  
- **Outputs**:  
  - `best_classifier` (model with optimal hyperparams)  
  - `calibrated_classifier`  
  - `training_time`

### `label_prediction_eval`
- **Purpose**: Predict test-set labels and evaluate performance metrics.  
- **Functions**:
  - `label_prediction()`:  
    - **Inputs**: `expression_matrix`, `prediction_technique`, `voting_type`, `trained_classifier`, `calibrated_classifier`, `X_test`, `test_sample_names`, `validation_sample_names2`, `classifier_type`, `target_column_name`  
    - **Outputs**: `y_pred`, `test_time`  
  - `classifier_evaluation()`:  
    - **Inputs**: `y_test`, `train_time`, `classifier_type`, `expression_matrix`, `voting_type`, `prediction_technique`, `trained_classifier`, `calibrated_classifier`, `test_sample_names`, `X_test`, `validation_sample_names2`, `target_column_name`, `output_file='OUTPUT_FILE'`  
    - **Output**: Evaluation metrics stored into `OUTPUT_FILE`

---

## Notes
- The pipeline is designed for **Alzheimer’s disease** detection using different ML models.  
- Ensure the environment includes required packages (e.g., `scikit-learn`, `pytorch`, etc.).  
- Hardcoded sample lists remain consistent across all model experiments for fair comparisons.

---

**End of Document**  
Master’s Thesis — Nazlıgül Keske  
Saarland University, Center for Bioinformatics  
