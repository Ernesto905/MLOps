from zenml import step 
import logging
import pandas as pd 

@step 
def model_train(df : pd.DataFrame) -> None:
    """
    trains model based on ingested data

    Args:
        df: ingested data
    """
    pass

