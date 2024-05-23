import pandas as pd
import logging
from zenml import step

from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy
from typing_extensions import Annotated
from typing import Tuple 


@step 
def clean_data(df : pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.DataFrame, "y_train"],
    Annotated[pd.DataFrame, "y_test"],
]:   
    """
    Cleans up our ingested data and divides it into training and testing sets.

    Args:
        df: Ingested data 
    Returns:
        pd.DataFrame: Pandas dataframe of cleaned data
    """
    try:
        process_strategy = DataPreProcessStrategy()
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(df, process_strategy)

        processed_data = data_cleaning.handle_data() 
        data_cleaning = DataCleaning(processed_data, divide_strategy)

        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed")
    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e

