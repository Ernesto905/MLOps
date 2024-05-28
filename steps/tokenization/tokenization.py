import pandas as pd
import logging
from zenml import step
from datasets import load_dataset, Dataset

from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreprocessStrategy
from typing_extensions import Annotated
from typing import Tuple 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


@step 
def tokenization(train_dataset: Dataset, eval_dataset: Dataset) -> Tuple[
    Annotated[Dataset, "train_dataset"],
    Annotated[Dataset, "eval_dataset"],
    ]:   
    """
    Tokenizes our training and evaluation datasets

    Args:
        train_dataset: Ingested training data 
        eval_dataset: Ingested evaluation data 
    Returns:
        Dataset []: An annotated tuple of our training and evaluation dataset. Now tokenized 
    """
    try:
        # Tokenize our tweets
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        def tokenize_data(examples):
            return tokenizer(examples["Tweet"], padding="max_length", truncation=True, max_length=128)

        train_dataset = train_dataset.map(tokenize_data, batched=True)
        eval_dataset = eval_dataset.map(tokenize_data, batched=True)

        return train_dataset, eval_dataset

    except Exception as e:
        logging.error("Error in tokenizing data: {}".format(e))
        raise e

