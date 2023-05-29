"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import make_predictions, report_accuracy, split_data, create_confusion_matrix, create_confusion_matrix_mlflow


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["example_iris_data", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split",
            ),
            node(
                func=make_predictions,
                inputs=["X_train", "X_test", "y_train"],
                outputs="y_pred",
                name="make_predictions",
            ),
            node(
                func=report_accuracy,
                inputs=["y_pred", "y_test"],
                outputs="metrics",
                name="report_accuracy",
            ),
            node(
                func=create_confusion_matrix,
                inputs=["y_pred", "y_test"],
                outputs="confusion_matrix",
                name="create_confusion_matrix",
            ),
            node(
                func=create_confusion_matrix_mlflow,
                inputs=["y_pred", "y_test"],
                outputs="confusion_matrix_mlf",
                name="create_confusion_matrix_mlflow",
            ),
        ]
    )
