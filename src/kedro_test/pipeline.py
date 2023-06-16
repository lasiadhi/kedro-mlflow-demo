"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_training, report_accuracy, split_data, create_confusion_matrix, make_predictions


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["example_iris_data", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data",
            ),
            node(
                func=model_training,
                inputs=["X_train", "y_train"],
                outputs="trained_model",
                name="model_training",
            ),
            node(
                func=make_predictions,
                inputs=["trained_model", "X_test"],
                outputs="y_pred",
                name="make_predictions",
            ),
            node(
                func=report_accuracy,
                inputs=["y_pred", "y_test"],
                outputs=None,
                name="report_accuracy",
            ),
            node(
                func=create_confusion_matrix,
                inputs=["y_pred", "y_test"],
                outputs="confusion_matrix",
                name="create_confusion_matrix",
            ),
        ]
    )
