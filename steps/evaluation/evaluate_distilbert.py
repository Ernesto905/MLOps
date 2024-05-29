from zenml import step
import logging
import pandas as pd 
from typing_extensions import Annotated
    
import mlflow
from zenml.client import Client
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_distilbert(model_path: str, eval_dataset: Dataset) -> Annotated[dict, "eval_scores"]:    
    """
    Evaluates our model

    Args:
        model: Our ingested data
    Returns:
        {}: A dictionary specifying the evaluation scores for our model
    """

    try:
        trainer = Trainer(
            model=AutoModelForSequenceClassification.from_pretrained(model_path),
            eval_dataset=eval_dataset
        )
        scores = trainer.evaluate()
        return scores
    except Exception as e:
        logging.error("Error evaluating model: {e}".format(e))
        raise e
