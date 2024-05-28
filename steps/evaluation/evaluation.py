from zenml import step 
import logging
import pandas as pd 
from src.evaluation import MSE, RMSE, R2
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
    
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
        model: RegressorMixin,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
    ) -> Tuple[
        Annotated[float, "r2_score"],
        Annotated[float, "rmse"],
    ]:
    """
    Evaluates our model

    Args:
        df: Our ingested data
    Returns:
        
    """

    try:
        prediction = model.predict(X_test)
        
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)


        # Log metrics with MLFlow
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        return r2, rmse
    except Exception as e:
        logging.error("Error evaluating model: {e}".format(e))
        raise e
