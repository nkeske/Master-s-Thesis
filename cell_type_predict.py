import pandas as pd
from sklearn.metrics import accuracy_score
from collections import defaultdict


# Function to reach cell-type prectictivity weights
def cell_type_weights(validation_sample_names2, trained_classifier, expression_matrix, test_sample_names,
                      classifier_type):
    donor_wise_data = expression_matrix.groupby('Donor ID')
    cell_type_accuracies = []
    test_accuracy_weights = {}
    test_accuracy_weights_transformed = {}

    # Donor ID belong to validation set 2 was utilized to find cell-type based average prediction accuracies
    for donor_id in validation_sample_names2:

        donor_data = donor_wise_data.get_group(donor_id)
        donor_data_copy = donor_data.copy()
        donor_data.drop('Donor ID', axis=1, inplace=True)
        donor_data.drop('Cognitive Status', axis=1, inplace=True)
        donor_subclass_data = donor_data['Subclass']
        donor_data.drop('Subclass', axis=1, inplace=True)

        donor_predictions = trained_classifier.predict(donor_data)
        predictions_with_subclass = dict(zip(donor_subclass_data.index, donor_predictions))

        # Prediction accuracies were accompanied by corresponding cell-type
        predictions_with_subclass = pd.Series(predictions_with_subclass)
        subclass_grouped_predictions = predictions_with_subclass.groupby(donor_subclass_data)

        accuracies = {}
        # Iterated over each subclass to reach that subclasses predictions.
        for subclass, predictions in subclass_grouped_predictions:

            group_labels = donor_data_copy[donor_data_copy['Subclass'] == subclass]['Cognitive Status']
            if isinstance(predictions.iloc[0], list):
                predictions = predictions.apply(lambda x: x[0])
            if classifier_type == 'MLP':
                predictions = [item for sublist in predictions for item in sublist]

            # Accuracies of each subclass were calculated and added to dictionary
            subclass_accuracy = accuracy_score(group_labels, predictions)
            accuracies[subclass] = subclass_accuracy

        cell_type_accuracies.append(accuracies)

    mean_accuracy_dict = defaultdict(lambda: {'sum': 0, 'count': 0})

    # Finding the average accuracies
    for d in cell_type_accuracies:
        for cell_type, value in d.items():
            mean_accuracy_dict[cell_type]['sum'] += value
            mean_accuracy_dict[cell_type]['count'] += 1

    # Calculating the mean for each cell type
    mean_accuracies = {cell_type: accuracy_info['sum'] / accuracy_info['count'] for cell_type, accuracy_info in
                       mean_accuracy_dict.items()}

    # Sorting the results to find top-six cell-types that has highest prediction accuracies
    sorted_mean_result = dict(sorted(mean_accuracies.items(), key=lambda item: item[1], reverse=True))
    top_cell_types = list(sorted_mean_result.keys())[:6]

    # Cubic transformation of the cell-type accuracy weights
    transformed_sorted_mean_result = {key: value ** 3 for key, value in sorted_mean_result.items()}

    # Min-max normalization on the transformed accuracies to carry them to the same range
    min_value = min(transformed_sorted_mean_result.values())
    max_value = max(transformed_sorted_mean_result.values())

    normalized_transformed_sorted_mean_result = {key: (value - min_value) / (max_value - min_value) for key, value in
                                                 transformed_sorted_mean_result.items()}

    # Creating the test_accuracy_weights dictionary to use in acrual test-set classification.
    for donor_id in test_sample_names:
        donor_data = donor_wise_data.get_group(donor_id)
        mean_list = {}
        mean_list_transformed = {}
        subclass_group_labels = donor_data.groupby('Subclass')
        for subclass_name, data in subclass_group_labels:
            weight_val = sorted_mean_result[subclass_name]
            mean_list[subclass_name] = weight_val

            weight_val_transformed = normalized_transformed_sorted_mean_result[subclass_name]
            mean_list_transformed[subclass_name] = weight_val_transformed

        test_accuracy_weights[donor_id] = mean_list
        test_accuracy_weights_transformed[donor_id] = mean_list_transformed

    return top_cell_types, test_accuracy_weights, test_accuracy_weights_transformed


# Function for filtering single-cell dataset for the application of Majority Voting with 6-most predictive cell types

def top6_testset_singlecell(expression_matrix, test_sample_names, target_column_name, top_subclass):
    test_data_list = []
    test_metadata_list = []
    test_sample_labels = []
    test_group_names = []

    donor_wise_data = expression_matrix.groupby('Donor ID')
    for group_name in test_sample_names:
        donor_data = donor_wise_data.get_group(group_name)
        # Filtering the data to just include cell-types withing top-6 most predictive cell-types
        donor_data = donor_data[donor_data['Subclass'].isin(top_subclass)]
        disease_by_cell = donor_data[target_column_name]
        disease_by_groupname = donor_data[target_column_name][0]
        donor_info = donor_data['Donor ID']

        donor_data.drop(['Donor ID', target_column_name], axis=1, inplace=True)
        test_metadata_list.append(disease_by_cell)
        donor_data['Donor ID'] = donor_info.values
        test_data_list.append(donor_data)
        test_sample_labels.append(disease_by_groupname)

    test_data = pd.concat(test_data_list, ignore_index=True)
    test_metadata = pd.concat(test_metadata_list, ignore_index=True)

    return test_data, test_metadata, test_sample_labels
