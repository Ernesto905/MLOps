from zenml import step 
import logging
import pandas as pd 
    
import mlflow
from zenml.client import Client
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_distilbert(model: Trainer) -> {}:
    """
    Evaluates our model

    Args:
        model: Our ingested data
    Returns:
        {}: A dictionary specifying the evaluation scores for our model
    """

    try:
        scores = model.evaluate()
        return scores
    except Exception as e:
        logging.error("Error evaluating model: {e}".format(e))
        raise e
