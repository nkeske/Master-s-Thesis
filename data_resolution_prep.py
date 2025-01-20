import pandas as pd


def bulk_sample_prep(expression, train_sample_names, test_sample_names, validation_sample_names1,
                     validation_sample_names2,
                     target_column_name):
    # Creating the bulk expression matrix by grouping the data by Donor ID
    expression = expression.drop('Subclass', axis=1)
    expression_matrix = expression.iloc[:, :-1]
    expression_by_sample = expression_matrix.groupby('Donor ID').mean()
    disease_by_sample = expression.groupby('Donor ID')[target_column_name].first()

    # Creating training expression matrix and label data
    X_train = expression_by_sample.loc[expression_by_sample.index.isin(train_sample_names)]
    y_train = disease_by_sample.loc[disease_by_sample.index.isin(train_sample_names)]
    X_train = X_train.dropna()
    y_train = y_train.loc[X_train.index]

    # Creating testing expression matrix and label data
    X_test = expression_by_sample.loc[expression_by_sample.index.isin(test_sample_names)]
    y_test = disease_by_sample.loc[disease_by_sample.index.isin(test_sample_names)]
    X_test = X_test.dropna()
    y_test = y_test.loc[X_test.index]

    # Creating validation expression matrix and label data
    X_val1 = expression_by_sample.loc[expression_by_sample.index.isin(validation_sample_names1)]
    y_val1 = disease_by_sample.loc[disease_by_sample.index.isin(validation_sample_names1)]
    X_val1 = X_val1.dropna()
    y_val1 = y_val1.loc[X_val1.index]

    X_val2 = expression_by_sample.loc[expression_by_sample.index.isin(validation_sample_names2)]
    y_val2 = disease_by_sample.loc[disease_by_sample.index.isin(validation_sample_names2)]
    X_val2 = X_val2.dropna()
    y_val2 = y_val2.loc[X_val2.index]

    return X_train, y_train, X_test, y_test, X_val1, y_val1, X_val2, y_val2


def pseudobulk_sample_prep(expression_matrix, train_sample_names, test_sample_names, validation_sample_names1,
                           validation_sample_names2, target_column_name):
    expression_by_sample = expression_matrix.groupby('Donor ID')

    training_data_list = []
    training_metadata_list = []
    validation_data_list1 = []
    validation_metadata_list1 = []
    validation_data_list2 = []
    validation_metadata_list2 = []
    test_data_list = []
    test_metadata_list = []
    train_sample_labels = []
    test_sample_labels = []
    val_sample_labels1 = []
    val_sample_labels2 = []
    test_group_names = []

    for group_name in train_sample_names:
        filtered_group = expression_by_sample.get_group(group_name)
        # Getting sample wise disease status
        disease_by_groupname = filtered_group[target_column_name][0]
        # Drop unnecessary Donor ID data column
        filtered_group.drop('Donor ID', axis=1, inplace=True)
        # Take the disease status based on the each cell type on the corresponding sample's data
        disease_by_celltype = filtered_group.groupby('Subclass')[target_column_name].first()
        # Drop unnecessary Cognitive Status data column
        filtered_group.drop(target_column_name, axis=1, inplace=True)
        # Generate a gene expression matrix taking mean gene expression for each cell type groups
        grouped_mean_expression = filtered_group.groupby('Subclass').mean()

        training_metadata_list.append(disease_by_celltype)
        training_data_list.append(grouped_mean_expression)
        train_sample_labels.append(disease_by_groupname)

    for group_name in test_sample_names:
        test_group_names.append(group_name)
        filtered_group = expression_by_sample.get_group(group_name)
        disease_by_groupname = filtered_group[target_column_name][0]
        # Drop unnecessary Donor ID data column
        filtered_group.drop('Donor ID', axis=1, inplace=True)
        grouped_data = filtered_group.groupby('Subclass')
        # Take the disease status based on the each cell type on the corresponding sample's data
        disease_by_celltype = grouped_data[target_column_name].first()
        subclasses = grouped_data.groups.keys()
        # Drop unnecessary Cognitive Status data column
        filtered_group.drop(target_column_name, axis=1, inplace=True)
        # Generate a gene expression matrix taking mean gene expression for each cell type groups
        grouped_mean_expression = filtered_group.groupby('Subclass').mean()

        test_metadata_list.append(disease_by_celltype)
        grouped_mean_expression['Subclass'] = subclasses
        grouped_mean_expression['Donor ID'] = group_name

        test_data_list.append(grouped_mean_expression)
        test_sample_labels.append(disease_by_groupname)

    for group_name in validation_sample_names1:
        filtered_group = expression_by_sample.get_group(group_name)
        # Getting sample wise disease status
        disease_by_groupname = filtered_group[target_column_name][0]
        # Drop unnecessary Donor ID data column
        filtered_group.drop('Donor ID', axis=1, inplace=True)
        # Take the disease status based on the each cell type on the corresponding sample's data
        disease_by_celltype = filtered_group.groupby('Subclass')[target_column_name].first()
        # Drop unnecessary Cognitive Status data column
        filtered_group.drop(target_column_name, axis=1, inplace=True)
        # Generate a gene expression matrix taking mean gene expression for each cell type groups
        grouped_mean_expression = filtered_group.groupby('Subclass').mean()

        validation_metadata_list1.append(disease_by_celltype)
        validation_data_list1.append(grouped_mean_expression)
        val_sample_labels1.append(disease_by_groupname)

    for group_name in validation_sample_names2:
        filtered_group = expression_by_sample.get_group(group_name)
        # Getting sample wise disease status
        disease_by_groupname = filtered_group[target_column_name][0]
        # Drop unnecessary Donor ID data column
        filtered_group.drop('Donor ID', axis=1, inplace=True)
        # Take the disease status based on the each cell type on the corresponding sample's data
        disease_by_celltype = filtered_group.groupby('Subclass')[target_column_name].first()
        # Drop unnecessary Cognitive Status data column
        filtered_group.drop(target_column_name, axis=1, inplace=True)
        # Generate a gene expression matrix taking mean gene expression for each cell type groups
        grouped_mean_expression = filtered_group.groupby('Subclass').mean()

        validation_metadata_list2.append(disease_by_celltype)
        validation_data_list2.append(grouped_mean_expression)
        val_sample_labels2.append(disease_by_groupname)

    # Gather the data lists in a dataframe to have whole train, test and validation sets
    training_data = pd.concat(training_data_list, ignore_index=True)
    training_metadata = pd.concat(training_metadata_list, ignore_index=True)

    test_data = pd.concat(test_data_list, ignore_index=True)
    test_metadata = pd.concat(test_metadata_list, ignore_index=True)

    validation_data1 = pd.concat(validation_data_list1, ignore_index=True)
    validation_metadata1 = pd.concat(validation_metadata_list1, ignore_index=True)

    validation_data2 = pd.concat(validation_data_list2, ignore_index=True)
    validation_metadata2 = pd.concat(validation_metadata_list2, ignore_index=True)

    # Get rid of NA values resulted from cell type based mean calculation
    training_data = training_data.dropna()
    training_metadata = training_metadata.loc[training_data.index]

    test_data = test_data.dropna()
    test_metadata = test_metadata.loc[test_data.index]

    validation_data1 = validation_data1.dropna()
    validation_metadata1 = validation_metadata1.loc[validation_data1.index]

    validation_data2 = validation_data2.dropna()
    validation_metadata2 = validation_metadata2.loc[validation_data2.index]

    return (training_data, training_metadata, test_data, test_metadata, validation_data1, validation_metadata1,
            validation_data2, validation_metadata2, train_sample_labels, test_sample_labels, val_sample_labels1,
            val_sample_labels2)


def singlecell_sample_prep(expression_matrix, train_sample_names, test_sample_names, validation_sample_names1,
                           validation_sample_names2,
                           target_column_name):
    expression_by_sample = expression_matrix.groupby('Donor ID')

    training_data_list = []
    training_metadata_list = []
    test_data_list = []
    test_metadata_list = []
    validation_data_list1 = []
    validation_metadata_list1 = []
    validation_data_list2 = []
    validation_metadata_list2 = []
    test_sample_labels = []

    for group_name in train_sample_names + test_sample_names + validation_sample_names1 + validation_sample_names2:
        filtered_group = expression_by_sample.get_group(group_name)
        disease_by_cell = filtered_group[target_column_name]
        disease_by_groupname = filtered_group[target_column_name][0]
        donor_info = filtered_group['Donor ID']

        if group_name in train_sample_names:
            filtered_group.drop(['Donor ID', target_column_name, 'Subclass'], axis=1, inplace=True)
            training_metadata_list.append(disease_by_cell)
            training_data_list.append(filtered_group)
        elif group_name in test_sample_names:
            filtered_group.drop(['Donor ID', target_column_name], axis=1, inplace=True)
            test_metadata_list.append(disease_by_cell)
            filtered_group['Donor ID'] = donor_info.values
            test_data_list.append(filtered_group)
            test_sample_labels.append(disease_by_groupname)
        elif group_name in validation_sample_names1:
            filtered_group.drop(['Donor ID', target_column_name, 'Subclass'], axis=1, inplace=True)
            validation_metadata_list1.append(disease_by_cell)
            validation_data_list1.append(filtered_group)
        elif group_name in validation_sample_names2:
            filtered_group.drop(['Donor ID', target_column_name, 'Subclass'], axis=1, inplace=True)
            validation_metadata_list2.append(disease_by_cell)
            validation_data_list2.append(filtered_group)

    training_data = pd.concat(training_data_list, ignore_index=True)
    training_metadata = pd.concat(training_metadata_list, ignore_index=True)
    test_data = pd.concat(test_data_list, ignore_index=True)
    test_metadata = pd.concat(test_metadata_list, ignore_index=True)
    validation_data1 = pd.concat(validation_data_list1, ignore_index=True)
    validation_metadata1 = pd.concat(validation_metadata_list1, ignore_index=True)
    validation_data2 = pd.concat(validation_data_list2, ignore_index=True)
    validation_metadata2 = pd.concat(validation_metadata_list2, ignore_index=True)

    return (training_data, training_metadata, test_data, test_metadata,
            validation_data1, validation_metadata1, validation_data2, validation_metadata2, test_sample_labels)
