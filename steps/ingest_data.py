import logging 
import pandas as pd 
from zenml import step 

class IngestData:
    """
    Ingesting data from a datapath
    """
    def __init__(self, data_path: str):
        self.data_path = data_path 

    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)

@step 
def ingest_data(data_path : str) -> pd.DataFrame:
    """
    Ingesting the datapath 

    Args:
        data_path: path to the data 
    Returns:
        pd.Dataframe: the ingested data
    """
    try:
        ingest_data = IngestData(data_path)
        data = ingest_data.get_data()
        print("AT THIS POINT DATA COLUMNS ARE:")

        for col in data.columns:
            print(col)
        return data 
        
    except Exception as e:
        logging.error("Error while ingesting data: {}".format(e))
        raise e


