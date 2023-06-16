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


def model_training(
    X_train: pd.DataFrame, y_train: pd.Series
) -> sklearn:
    """Uses 1-nearest neighbour classifier to create predictions.

    Args:
        X_train: Training data of features.
        y_train: Training data for target.

    Returns:
        sklearn: Trained LR model
    """

    X_train_numpy = X_train.to_numpy()
    
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.01, 0.05, 0.1, 1.0, 10.0],  # Regularization parameter
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

    sklearn.log_model(sk_model=best_model, artifact_path="LR_CV_model")
    mlflow.log_param("Best hyperparameters", best_params)

    return best_model

def make_predictions(trained_model: sklearn, X_test: pd.DataFrame)-> pd.Series:
    """_summary_

    Args:
        trained_model (_type_): _description_
        X_test (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    X_test_numpy = X_test.to_numpy()

    # Make predictions on the test set using the trained model
    y_pred = trained_model.predict(X_test_numpy)

    return y_pred


def report_accuracy(y_pred: pd.Series, y_test: pd.Series):
    """Calculates and logs the accuracy.

    Args:
        y_pred: Predicted target.
        y_test: True target.
    """

    accuracy = (y_pred == y_test).sum() / len(y_test)

    logger = logging.getLogger(__name__)
    logger.info("Model has accuracy of %.3f on test data.", accuracy)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("time of prediction", str(datetime.now()))
    mlflow.set_tag("Model type", "only_with_vitals")

def create_confusion_matrix(y_pred: pd.Series, y_test: pd.Series):
    """_summary_

    Args:
        y_pred (pd.Series): _description_
        y_test (pd.Series): _description_
    """
    confusion_matrix = pd.crosstab(
        y_test, y_pred, rownames=["Actual"], colnames=["Predicted"]
    )
    fig, _ = plt.subplots(1, 1, figsize=(6,6))
    sn.heatmap(confusion_matrix, annot=True)
    mlflow.log_figure(fig, "plots/cm_plot.png") ## Save inside mlflow artifacts
    return fig