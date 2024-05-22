import pandas as pd
import logging
from zenml import step

@step 
def clean_data(df : pd.DataFrame) -> pd.DataFrame:   
    """
    Cleans up our ingested data 

    Args:
        df: Ingested data 
    Returns:
        pd.DataFrame: Pandas dataframe of cleaned data
    """
    return df

