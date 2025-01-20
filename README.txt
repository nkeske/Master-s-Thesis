Nazlıgül Keske
7025902
Saarland University
Center for Bioinformatics
Master's Thesis

Program applies differet machine learning methods (Decision trees, Random forests, Support vector machines with several different
kernel function and one neural network method Multi-layer perceptirons to different resolution (bulk, pseudo-bulk and single-cell)
brain data for thr prediction of Alzheimer's disease.

To process the whole pipeline 'pipeline_run.ipynb' should runned within the repository containing:
	- train_sample_names
	- test_sample_names
	- validation_sample_names1
	- validation_sample_names2
	(These sets are created beforehand as explained at 'Sample-Splitting' section of the thesis. Since the same sets used to
	assess all different models, they are hardcoded as a text file.)
	- 'SEAAD_MTG_RNAseq.h5ad'
	- Variance_Estimation_2000.h5ad 
	(This data containes gene expression data. Highly variable genes estimated and filtered accordingly and saved 
	in this file. The hihly variable genes are obtained with 'gene_variance_estimation.py')

Modules:
	* data_loading_processing : Loads the anndata and sample set name files to be usable in upcoming functions
		inputs: expression_file, train_sample_file, test_sample_file, validation_set1_file, validation_set2_file, target_column_name
		outputs: expression_matrix, train_sample_names, test_sample_names, validation_sample_names1, validation_sample_names2
	
	* data_resolution_prep: It tooks the load expression matrix and creates bulk, pseudo-bulk and single cell resolutions with sub-functions:
		* bulk_sample_prep(), pseudobulk_sample_prep() and singlecell_sample_prep()
		inputs: expression_matrix, train_sample_names, test_sample_names, validation_sample_names1,validation_sample_names2, target_column_name
		outputs: training_data, training_metadata, test_data, test_metadata, validation_data1, validation_metadata1, validation_data2, validation_metadata2, test_sample_labels
	
	*gene_variance_estimation: Applies variance estimation analysis to SEA-AD 'SEAAD_MTG_RNAseq.h5ad' dataset and 
		returns filtered expression matrix 'Variance_Estimation_2000_full.h5ad' and stores it to repository.
		inputs: expression_file, train_sample_file
			 
		
	* optimization: Consist of seperate hyperparameter optimization for each different model. The optimization
		function is checked and applied according to classifier type.
		inputs: X_train, y_train, X_val, y_val, classifier_type
		outputs: best_classifier, calibrated_classifier, training_time

	* label_prediction_eval: This is the function where test set sample labels were predicted. After prediction, evaluation
		metrices were calculated then saved to 'OUTPUT_FILE'.
		label_prediction(): 
			inputs: expression_matrix, prediction_technique, voting_type, trained_classifier, calibrated_classifier,
                     		X_test, test_sample_names, validation_sample_names2, classifier_type, target_column_name
			outputs: y_pred, test_time
		classifier_evaluation(): 
			inputs: y_test, train_time, classifier_type, expression_matrix, voting_type, prediction_technique,
                          trained_classifier, calibrated_classifier, test_sample_names, X_test,
                          validation_sample_names2, target_column_name, output_file='OUTPUT_FILE'





 
