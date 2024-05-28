from zenml import step 
import logging
import pandas as pd 

from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin 
from ..config import ModelNameConfig

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
    ) -> RegressorMixin:
    """
    trains model based on ingested data

    Args:
        X_train: Our ingested traning data 
        X_test: Our ingested testing data 
        y_train: Our training labels 
        y_test: Our testing targets
    Returns: 
        RegressorMixin: 
    """

    try:
        model = None

        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog() # Automatically logs our model on our MLFlow dashboard
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        # elif config.model_name == "KMeansClustering":
        #     model = ...
        #     trained_model = ...
        #     return trained_model
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        raise e

