import logging 
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class defining evaluation strategy for our models
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray):
        """
        Calculates the scores for the model  

        Args:
            y_true: true labels
            y_pred: predicted labels
        Returns:
            None
        """
        pass

class MSE(Evaluation):
    """
    Evaluation strategy using mean squared error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates scores for the model using mean squared error
        
        Args:
            y_true: true labels
            y_pred: predicted labels
        Returns:
            None
        """
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e

class R2(Evaluation):
    """
    Evaluation strategy using R2 Score
    """ 

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates scores for the model using the R2 method 
        
        Args:
            y_true: true labels
            y_pred: predicted labels
        Returns:
            None
        """
        try:
            logging.info("Calculating R2")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("R2: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating R2: {}".format(e))
            raise e

class RMSE(Evaluation):
    """
    Evaluation strategy uses the root mean squared error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates scores for the model using the the root mean squared error        

        Args:
            y_true: true labels
            y_pred: predicted labels
        Returns:
            None
        """
        try:
            logging.info("Calculating RMSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("RMSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating RMSE: {}".format(e))
            raise e
