from zenml import step 
import logging
import pandas as pd 


@step 
def evaluate_model(df: pd.DataFrame) -> None:
    """
    Evaluates our model

    Args:
        df: Our ingested data
    """
    pass
