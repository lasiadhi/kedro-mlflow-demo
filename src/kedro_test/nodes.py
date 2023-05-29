"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from kedro_mlflow.io.metrics import MlflowMetricDataSet
from kedro_mlflow.io.artifacts import MlflowArtifactDataSet
import matplotlib.pyplot as plt
import seaborn as sn
import mlflow
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from mlflow import sklearn
from sklearn.model_selection import GridSearchCV


def split_data(
    data: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into features and target training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """

    data_train = data.sample(
        frac=parameters["train_fraction"], random_state=parameters["random_state"]
    )
    data_test = data.drop(data_train.index)

    X_train = data_train.drop(columns=parameters["target_column"])
    X_test = data_test.drop(columns=parameters["target_column"])
    y_train = data_train[parameters["target_column"]]
    y_test = data_test[parameters["target_column"]]

    return X_train, X_test, y_train, y_test


def make_predictions(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series
) -> pd.Series:
    """Uses 1-nearest neighbour classifier to create predictions.

    Args:
        X_train: Training data of features.
        y_train: Training data for target.
        X_test: Test data for features.

    Returns:
        y_pred: Prediction of the target variable.
    """

    X_train_numpy = X_train.to_numpy()
    X_test_numpy = X_test.to_numpy()

    #print('X_train_numpy', X_train_numpy)
    #print('y_train', y_train)

    """ 1-nearest neighbour
    squared_distances = np.sum(
        (X_train_numpy[:, None, :] - X_test_numpy[None, :, :]) ** 2, axis=-1
    )
    nearest_neighbour = squared_distances.argmin(axis=0)
    y_pred = y_train.iloc[nearest_neighbour]
    y_pred.index = X_test.index
    """
    
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],  # Regularization parameter
        'solver': ['liblinear', 'saga']  # Solver algorithm
    }
    # Create a logistic regression classifier
    clf = LogisticRegression()
    # Perform grid search for hyperparameter tuning
    grid_search = GridSearchCV(clf, param_grid, cv=5)
    grid_search.fit(X_train_numpy, y_train)

    # Retrieve the best hyperparameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Make predictions on the test set using the best model
    y_pred = best_model.predict(X_test_numpy)
    sklearn.log_model(sk_model=best_model, artifact_path="LR_CV_model")

    #print("Best hyperparameters:", best_params)
    mlflow.log_param("Best hyperparameters", best_params)
    return y_pred


def report_accuracy(y_pred: pd.Series, y_test: pd.Series):
    """Calculates and logs the accuracy.

    Args:
        y_pred: Predicted target.
        y_test: True target.
    """
    #metric_ds = MlflowMetricDataSet(key="accuracy")

    accuracy = (y_pred == y_test).sum() / len(y_test)
    #metric_ds.save(accuracy)
    logger = logging.getLogger(__name__)
    logger.info("Model has accuracy of %.3f on test data.", accuracy)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("time of prediction", str(datetime.now()))
    mlflow.set_tag("Model Version", 200)
    return {"accuracy": accuracy} # for Kedro exp-tracking only

def create_confusion_matrix(y_pred: pd.Series, y_test: pd.Series):
    confusion_matrix = pd.crosstab(
        y_test, y_pred, rownames=["Actual"], colnames=["Predicted"]
    )
    sn.heatmap(confusion_matrix, annot=True)
    return plt # for Kedro exp-tracking only

def create_confusion_matrix_mlflow(y_pred: pd.Series, y_test: pd.Series):
    confusion_matrix = pd.crosstab(
        y_test, y_pred, rownames=["Actual"], colnames=["Predicted"]
    )
    fig, _ = plt.subplots(1, 1, figsize=(6,6))
    sn.heatmap(confusion_matrix, annot=True)
    mlflow.log_figure(fig, "plots/cm_plot.png") ## working. directly will be in mlflow artifacts
    return plt
