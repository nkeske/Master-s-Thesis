import pandas as pd
import numpy as np
import cell_type_predict
from time import process_time
from sklearn.metrics import confusion_matrix
import os


def label_prediction(expression_matrix, prediction_technique, voting_type, trained_classifier, calibrated_classifier,
                     X_test, test_sample_names, validation_sample_names2, classifier_type, target_column_name):
    prediction_times = []  # List to store prediction times
    y_pred = []  # Initialize y_pred

    # In bulk level, prediction applied directly
    if prediction_technique == 'Bulk':
        start_time = process_time()
        y_pred = trained_classifier.predict(X_test)
        end_time = process_time()
        prediction_times.append(end_time - start_time)
    else:
        # For Majority voting (MW), predictions should applied and voted in donor-wise
        if voting_type == 'Majority Voting':
            donor_wise_data = X_test.groupby('Donor ID')
            for donor_id, group_data in donor_wise_data:
                if donor_id in test_sample_names:
                    group_data = group_data.drop(columns=['Subclass', 'Donor ID'])
                    step_start_time = process_time()
                    predicted_labels = trained_classifier.predict(group_data)
                    step_end_time = process_time()

                    # MLP requires further adjustment in prediction array before voting applied
                    if classifier_type == 'MLP':
                        predicted_labels = [item for sublist in predicted_labels for item in sublist]
                        group_vote = max(set(predicted_labels), key=predicted_labels.count)
                    else:
                        group_vote = max(set(predicted_labels), key=predicted_labels.tolist().count)

                    y_pred.append(group_vote)
                    prediction_times.append(step_end_time - step_start_time)
        # For other three voting methods, cell-type predictivity information should be obtained
        elif voting_type in ['Weighted Voting', 'Transformed Weighted Voting', 'Majority Voting Top6']:
            top_cell_types, test_accuracy_weights, test_accuracy_weights_transformed = cell_type_predict.cell_type_weights(
                validation_sample_names2, trained_classifier, expression_matrix, test_sample_names, classifier_type)

            if voting_type == 'Majority Voting Top6':
                if prediction_technique == 'Pseudobulk':
                    X_test = X_test
                elif prediction_technique == 'Single cell':
                    test_data, test_metadata, test_sample_labels = cell_type_predict.top6_testset_singlecell(
                        expression_matrix, test_sample_names, target_column_name, top_cell_types)
                    donor_wise_data = X_test.groupby('Donor ID')
                    for donor_id, group_data in donor_wise_data:
                        if donor_id in test_sample_names:
                            group_data = group_data.drop(columns=['Subclass', 'Donor ID'])
                            start_time = process_time()
                            predicted_labels = trained_classifier.predict(group_data)
                            end_time = process_time()
                            if classifier_type == 'MLP':
                                predicted_labels = [item for sublist in predicted_labels for item in sublist]
                                donor_vote = max(set(predicted_labels), key=predicted_labels.count)
                            else:
                                donor_vote = max(set(predicted_labels), key=predicted_labels.tolist().count)
                            y_pred.append(donor_vote)
                            prediction_times.append(end_time - start_time)

            # In the case of weighted votings, cell-type weights for each individual-cells should be obtained for corresponding ID.
            else:
                donor_wise_data = X_test.groupby('Donor ID')
                for donor_id, group_data in donor_wise_data:
                    if donor_id in test_sample_names:
                        donor_test_weights = test_accuracy_weights[donor_id]
                        donor_test_weights_transformed = test_accuracy_weights_transformed[donor_id]
                        weight_list = []
                        weights_list_transformed = []

                        for index, row in group_data.iterrows():
                            cell_type = row['Subclass']
                            weight = donor_test_weights.get(cell_type, 0)
                            weight_list.append(weight)
                            accuracy_transformed = donor_test_weights_transformed.get(cell_type, 0)
                            weights_list_transformed.append(accuracy_transformed)
                        # Each individual cells' weights are stored in weights_array and weights_array_transformed.
                        weights_array = np.array(weight_list)
                        weights_array_transformed = np.array(weights_list_transformed)

                        group_data = group_data.drop(columns=['Subclass', 'Donor ID'])

                        if voting_type == 'Weighted Voting':
                            start_time = process_time()
                            # In order to predict class probabilities, 'calibrated_classifier' were used instead of 'trained_classifier'
                            initial_probabilities = calibrated_classifier.predict_proba(group_data)
                            end_time = process_time()

                            weighted_votes = np.zeros(initial_probabilities.shape[1])
                            # Weighted voting with call-type accuracies were applied (Explained in details in Chapter 4.4.2)
                            for i in range(len(weights_array)):
                                for j in range(initial_probabilities.shape[1]):
                                    if initial_probabilities[i, j] >= initial_probabilities[i, 1 - j]:
                                        weighted_votes[j] += initial_probabilities[i, j] * weights_array[i]
                                    else:
                                        weighted_votes[j] += initial_probabilities[i, j] * (
                                                1 - weights_array[i])

                            donor_vote = np.argmax(weighted_votes)

                            # The type of the label were checked to map it correctly.
                            if target_column_name == 'Cognitive Status':
                                label_mapping = {0: 'Dementia', 1: 'No dementia'}
                                donor_vote = label_mapping[donor_vote]
                            elif target_column_name == 'Overall AD neuropathological Change':
                                label_mapping = {0: 'AD', 1: 'Not AD'}
                                donor_vote = label_mapping[donor_vote]
                            y_pred.append(donor_vote)
                            prediction_times.append(end_time - start_time)

                        elif voting_type == 'Transformed Weighted Voting':
                            start_time = process_time()
                            initial_probabilities = calibrated_classifier.predict_proba(group_data)
                            end_time = process_time()

                            weighted_votes = np.zeros(initial_probabilities.shape[1])

                            for i in range(len(weights_array)):
                                for j in range(initial_probabilities.shape[1]):
                                    if initial_probabilities[i, j] >= initial_probabilities[i, 1 - j]:
                                        weighted_votes[j] += initial_probabilities[i, j] * \
                                                             weights_array_transformed[i]
                                    else:
                                        weighted_votes[j] += initial_probabilities[i, j] * (
                                                1 - weights_array_transformed[i])

                            donor_vote = np.argmax(weighted_votes)

                            if target_column_name == 'Cognitive Status':
                                label_mapping = {0: 'Dementia', 1: 'No dementia'}
                                donor_vote = label_mapping[donor_vote]
                            elif target_column_name == 'Overall AD neuropathological Change':
                                label_mapping = {0: 'AD', 1: 'Not AD'}
                                donor_vote = label_mapping[donor_vote]
                            y_pred.append(donor_vote)
                            prediction_times.append(end_time - start_time)

    test_time = sum(prediction_times)
    return y_pred, test_time


def classifier_evaluation(y_test, train_time, classifier_type, expression_matrix, voting_type, prediction_technique,
                          trained_classifier, calibrated_classifier, test_sample_names, X_test,
                          validation_sample_names2, target_column_name, output_file='OUTPUT_FILE'):
    # Label prediction function is called to predict labels and execution time.
    y_pred, execution_time = label_prediction(expression_matrix, prediction_technique, voting_type, trained_classifier,calibrated_classifier, X_test, test_sample_names, validation_sample_names2, classifier_type, target_column_name)

    y_pred_flat = list(y_pred)
    y_test_flat = list(y_test)

    # Confusion matrix is created for predicted and actual labels
    pred_confusion_matrix = confusion_matrix(y_test_flat, y_pred_flat)

    # The true positive, true negative, false positive, and false negative values were extracted from confusion matrix
    TN, FP, FN, TP = pred_confusion_matrix.ravel()

    # Ecaluation metrices were calculated
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1_score = 2 * TP / (2 * TP + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    # Dataframe were created to store all of the metric results and training and execution times
    results_df = pd.DataFrame({
        'Classifier': [classifier_type],
        'Voting Method': [voting_type],
        'Prediction Technique': [prediction_technique],
        'Accuracy': [accuracy],
        'F1 Score': [f1_score],
        'Sensitivity': [sensitivity],
        'Specificity': [specificity],
        'True Positives': [TP],
        'True Negatives': [TN],
        'False Positives': [FP],
        'False Negatives': [FN],
        'Training Time': [train_time],
        'Testing Time': [execution_time]
    })

    # The output repo is checked for output files existence
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        # Results were read and then updated version is written to the fike
        existing_results_df = pd.read_csv(output_file)
        updated_results_df = pd.concat([existing_results_df, results_df], ignore_index=True)
    else:
        updated_results_df = results_df
    updated_results_df.to_csv(output_file, index=False)


