from zenml import step 
import logging
import pandas as pd 

from src.mode_dev import LinearRegressionModel
from sklearn.base import RegressorMixin 
from .config import ModelNameConfig


@step 
def train_model(
    X_train: pd.dataframe,
    X_test: pd.dataframe,
    y_train: pd.dataframe,
    y_test: pd.dataframe,
    config: ModelNameConfig,
    ) -> RegressorMixin:
    """
    trains model based on ingested data

    Args:
        X_train: pd.dataframe
        X_test: pd.dataframe
        y_train: pd.dataframe
        y_test: pd.dataframe
    Returns: 
        RegressorMixin: 
    """

    try:
        model = None

        if config.model_name == "LinearRegression":
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

