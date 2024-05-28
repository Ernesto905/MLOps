from zenml import pipeline
from steps import (
    ingest_data,
    clean_dataset,
    tokenization,
    train_distilbert,
    evaluate_distilbert,
)
import logging 

@pipeline(enable_cache=True)
def sentiment_pipeline(data_path: str):

    data = ingest_data(data_path)

    train_dataset, eval_dataset = clean_dataset(data)
    train_dataset, eval_dataset = tokenization(train_dataset, eval_dataset)

    model = train_distilbert(train_dataset, eval_dataset)
    # eval_scores = evaluate_distilbert(model)
