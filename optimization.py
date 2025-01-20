from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from time import process_time
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier


def hyperparameter_optimization(X_train, y_train, X_val, y_val, classifier_type):

    def hyperparameter_op_dt(X_train, y_train, X_val, y_val, classifier_type):
        # Pipeline that includes normalization method and the classifier defined
        optimization_pipe = Pipeline([
            ('normalization', StandardScaler()),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])

        # Define the parameter grid with the hyperparameters to be optimized
        optimized_parameters = {
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_depth': [None, 1, 2, 3, 4, 5],
            'classifier__min_samples_split': [2, 3, 4, 5],

        }

        # Grid search is performed with 5-fold cross-validation method and with 50 parallel cores
        grid_search = GridSearchCV(optimization_pipe,  optimized_parameters, cv=5, n_jobs=50)
        # Search is applied for validation set 1
        grid_search.fit(X_val, y_val)

        #Best hyperparameters were extracted
        best_parameters = grid_search.best_params_
        best_criterion = best_parameters['classifier__criterion']
        best_max_depth = best_parameters['classifier__max_depth']
        best_min_samples_split = best_parameters['classifier__min_samples_split']

        #Best-classifier is generated with best parameters inluding pipeline
        best_classifier = DecisionTreeClassifier(
            criterion=best_criterion,
            max_depth=best_max_depth,
            min_samples_split=best_min_samples_split,
            random_state=42
        )

        best_classifier = Pipeline([
            ('normalization', StandardScaler()),
            ('classifier', best_classifier)
        ])
        # Fit the best classifier with training data. Process time is also acquired
        start_time = process_time()
        best_classifier.fit(X_train, y_train)
        end_time = process_time()
        execution_time = end_time - start_time

        #Since DT has predict_proba function, additional calibration is not necessary. (Calibration is applied for all
        # SVM's but not to the DT and RF and MLPs)
        calibrated_classifier = best_classifier

        #Function returns best classifier, calibrated classifier and execution time
        return best_classifier, calibrated_classifier, execution_time

    #Methods are applied following the same steps for evert classifier.
    def hyperparameter_op_rf(X_train, y_train, X_val, y_val, classifier_type):

        optimization_pipe = Pipeline([
            ('normalization', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ])

        optimized_parameters = {
            'classifier__n_estimators': [50, 100, 150],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 3, 4, 5],
            'classifier__min_samples_leaf': [1, 2, 3, 4, 5],
        }

        grid_search = GridSearchCV(optimization_pipe, optimized_parameters, cv=5, n_jobs=50)
        grid_search.fit(X_val, y_val)
        best_parameters = grid_search.best_params_

        best_n_estimators = best_parameters['classifier__n_estimators']
        best_max_depth = best_parameters['classifier__max_depth']
        best_min_samples_split = best_parameters['classifier__min_samples_split']
        best_min_samples_leaf = best_parameters['classifier__min_samples_leaf']

        best_classifier = RandomForestClassifier(
            n_estimators=best_n_estimators,
            max_depth=best_max_depth,
            min_samples_split=best_min_samples_split,
            min_samples_leaf=best_min_samples_leaf,
            random_state=42
        )
        best_classifier = Pipeline([
            ('normalization', StandardScaler()),
            ('classifier', best_classifier)
        ])
        start_time = process_time()
        best_classifier.fit(X_train, y_train)
        end_time = process_time()
        execution_time = end_time - start_time

        calibrated_classifier = best_classifier

        return best_classifier, calibrated_classifier, execution_time

    def hyperparameter_op_svm(X_train, y_train, X_val, y_val, classifier_type):

        optimization_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(random_state=42, probability=True))
        ])

        # For SVM's it additionally checks what is the type of SVM kernel. Based on that it optimizes certain parameters.
        if classifier_type == 'SVM linear':
            optimized_parameters = {
                'svm__kernel': ['linear'],
                'svm__C': [0.1, 1, 10, 100],
                'svm__gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto']
            }
        elif classifier_type == 'SVM polynomial':
            optimized_parameters = {
                'svm__kernel': ['poly'],
                'svm__C': [0.1, 1, 10, 100],
                'svm__gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
                'svm__degree': [2, 3, 4]
            }
        elif classifier_type == 'SVM radial basis function':
            optimized_parameters = {
                'svm__kernel': ['rbf'],
                'svm__C': [0.1, 1, 10, 100],
                'svm__gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto']
            }
        else:
            raise ValueError("Invalid classifier type.")

        grid_search = GridSearchCV(optimization_pipe, optimized_parameters, cv=5, n_jobs=50)
        grid_search.fit(X_val, y_val)
        best_parameters = grid_search.best_params_
        # Extract the best parameters for SVM
        overall_svm_parameters = {k.replace('svm__', ''): v for k, v in best_parameters.items() if k.startswith('svm__')}

        # Train the SVM classifier with the best hyperparameters on the training set
        best_classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(random_state=42, **overall_svm_parameters))
        ])

        #Calibration step is applied with CalibratedClassifierCV to be able to predict class probabilities in weighted voting.
        calibrated_classifier = CalibratedClassifierCV(best_classifier)
        calibrated_classifier.fit(X_train, y_train)

        start_time = process_time()
        best_classifier.fit(X_train, y_train)
        end_time = process_time()
        execution_time = end_time - start_time

        return best_classifier, calibrated_classifier, execution_time

    def hyperparameter_op_nusvm(X_train, y_train, X_val, y_val, classifier_type):

        optimization_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('nusvm', NuSVC(random_state=42))
        ])

        if classifier_type == 'NuSVM linear':
            optimized_parameters = {
                'nusvm__kernel': ['linear'],
                'nusvm__nu': [0.1, 0.5, 0.9],
                'nusvm__gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto']
            }
        elif classifier_type == 'NuSVM polynomial':
            optimized_parameters = {
                'nusvm__kernel': ['poly'],
                'nusvm__nu': [0.1, 0.5, 0.9],
                'nusvm__gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
                'nusvm__degree': [2, 3]
            }
        elif classifier_type == 'NuSVM radial basis function':
            optimized_parameters = {
                'nusvm__kernel': ['rbf'],
                'nusvm__nu': [0.1, 0.5, 0.9],
                'nusvm__gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto']
            }
        else:
            raise ValueError("Invalid classifier type")

        grid_search = GridSearchCV(optimization_pipe, optimized_parameters, cv=5, n_jobs=50)
        grid_search.fit(X_val, y_val)
        best_parameters = grid_search.best_params_

        overall_nusvm_parameters = {k.replace('nusvm__', ''): v for k, v in best_parameters.items() if k.startswith('nusvm__')}

        best_classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('nusvm', NuSVC(random_state=42, **overall_nusvm_parameters))
        ])

        #Calibration step is applied with CalibratedClassifierCV to be able to predict class probabilities in weighted voting.
        calibrated_classifier = CalibratedClassifierCV(best_classifier)
        calibrated_classifier.fit(X_train, y_train)

        start_time = process_time()
        best_classifier.fit(X_train, y_train)
        end_time = process_time()
        execution_time = end_time - start_time

        return best_classifier, calibrated_classifier, execution_time

    def hyperparameter_op_linearsvm(X_train, y_train, X_val, y_val, classifier_type):

        optimization_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('linearsvm', LinearSVC(random_state=42))
        ])

        # Define the parameter grid for hyperparameter optimization
        optimized_parameters = {
            'linearsvm__penalty': ['l1', 'l2'],
            'linearsvm__C': [0.01, 0.1, 1, 10, 100]
        }

        grid_search = GridSearchCV(optimization_pipe, optimized_parameters, cv=5, n_jobs=50)
        grid_search.fit(X_val, y_val)
        best_parameters = grid_search.best_params_

        overall_linearsvm_parameters = {k.replace('linearsvm__', ''): v for k, v in best_parameters.items() if
                            k.startswith('linearsvm__')}

        best_classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('linearsvm', LinearSVC(random_state=42, **overall_linearsvm_parameters))
        ])

        #Calibration step is applied with CalibratedClassifierCV to be able to predict class probabilities in weighted voting.
        calibrated_classifier = CalibratedClassifierCV(best_classifier)
        calibrated_classifier.fit(X_train, y_train)

        start_time = process_time()
        best_classifier.fit(X_train, y_train)
        end_time = process_time()
        execution_time = end_time - start_time

        return best_classifier, calibrated_classifier, execution_time

    #Function creates the initial MLP model before the application of optimization
    def create_mlp_model(hidden_layer_sizes=(100,), activation='relu', optimizer='adam', alpha=0.0001,
                         learning_rate=0.001):
        model = Sequential()
        for i, units in enumerate(hidden_layer_sizes):
            if i == 0:
                model.add(Dense(units, input_dim=X_train.shape[1], activation=activation))
            else:
                model.add(Dense(units, activation=activation))
        model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    #It creates Keras classifier from initially generated MLP model and allows us tp optimize parameters like epochs etc.
    def create_keras_classifier(hidden_layer_sizes=(100,), activation='relu', optimizer='adam', alpha=0.0001,
                                learning_rate=0.001, epochs=10, batch_size=32):
        return KerasClassifier(build_fn=create_mlp_model, epochs=epochs, batch_size=batch_size,
                               hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                               optimizer=optimizer, alpha=alpha, learning_rate=learning_rate, verbose=0)
    # Function that actually optimizes the MLP hyperparameters
    def hyperparameter_op_keras(X_train, y_train, X_val, y_val, classifier_type):
        input_dim = X_train.shape[1]
        num_classes = 2

        optimization_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', create_keras_classifier())
        ])

        optimized_parameters = {
            'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
            'classifier__activation': ['relu'],
            'classifier__optimizer': ['adam', 'sgd'],
            'classifier__alpha': [0.0001, 0.001, 0.01],
            'classifier__learning_rate': [0.001, 0.01],
            'classifier__epochs': [10, 20, 30, 40]
        }

        grid_search = GridSearchCV(optimization_pipe, optimized_parameters, cv=5, n_jobs=50)
        grid_search.fit(X_val, y_val)

        best_classifier = grid_search.best_estimator_
        best_parameters = grid_search.best_params_

        best_hidden_layer_sizes = best_parameters['classifier__hidden_layer_sizes']
        best_activation = best_parameters['classifier__activation']
        best_optimizer = best_parameters['classifier__optimizer']
        best_alpha = best_parameters['classifier__alpha']
        best_learning_rate = best_parameters['classifier__learning_rate']
        best_epochs = best_parameters['classifier__epochs']

        best_classifier_MLP = create_keras_classifier(hidden_layer_sizes=best_hidden_layer_sizes,
                                                      activation=best_activation,
                                                      optimizer=best_optimizer,
                                                      alpha=best_alpha,
                                                      learning_rate=best_learning_rate,
                                                      epochs=best_epochs)

        best_classifier_MLP  = Pipeline([
            ('normalization', StandardScaler()),
            ('classifier', best_classifier_MLP)
        ])

        start_time = process_time()
        best_classifier_MLP.fit(X_train, y_train)
        end_time = process_time()
        execution_time = end_time - start_time

        calibrated_classifier = best_classifier_MLP

        return best_classifier, calibrated_classifier, execution_time

    #Calling the hyperparameter optimization function according to entered classifier type
    optimization_function_call = {
        'Decision tree': hyperparameter_op_dt,
        'Random forest': hyperparameter_op_rf,
        'SVM linear': hyperparameter_op_svm,
        'SVM polynomial': hyperparameter_op_svm,
        'SVM radial basis function': hyperparameter_op_svm,
        'NuSVM linear': hyperparameter_op_nusvm,
        'NuSVM polynomial': hyperparameter_op_nusvm,
        'NuSVM radial basis function': hyperparameter_op_nusvm,
        'LinearSVC': hyperparameter_op_linearsvm,
        'MLP': hyperparameter_op_keras
    }
    if classifier_type in optimization_function_call.keys():
        best_classifier, calibrated_classifier, training_time = optimization_function_call[classifier_type](X_train, y_train,
                                                                                                      X_val, y_val,
                                                                                                      classifier_type)
    else:
        raise ValueError(f"Invalid classifier type: {classifier_type}")

    return best_classifier, calibrated_classifier, training_time
