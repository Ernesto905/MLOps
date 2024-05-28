import pandas as pd
import logging
from zenml import step
from datasets import load_dataset, Dataset

from typing_extensions import Annotated
from typing import Tuple 
from sklearn.model_selection import train_test_split


@step 
def clean_dataset(df: pd.DataFrame) -> Tuple[
    Annotated[Dataset, "train_dataset"],
    Annotated[Dataset, "eval_dataset"],
    ]:   
    """
    Cleans up our ingested data and divides it into training and evaluation datasets.

    Args:
        df: Ingested data 
    Returns:
        pd.DataFrame: Pandas dataframe of cleaned data
    """
    try:
        df = df.head(500) # Just use subset of our data for now 
        #Rename columns to something more intitive and drop the irrelevant ones 
        df = df.drop(['Borderlands', '2401'], axis=1)
        df = df.rename(columns={"Positive" : "label", "im getting on borderlands and i will murder you all ," : "Tweet"})


        # Convert all tweets to strings
        df['Tweet'] = df['Tweet'].astype(str)

        # Clean up labels and perform one hot encoding
        df = df[df['label'] != 'Irrelevant']

        sentiment_mapping = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
        df['label'] = df['label'].map(sentiment_mapping)


        train, eval_set = train_test_split(df, test_size=0.75, random_state=8, stratify=df['label'])

        # Load unto a Dataset
        train_dataset = Dataset.from_pandas(train)
        eval_dataset = Dataset.from_pandas(eval_set)

        return train_dataset, eval_dataset

    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e

