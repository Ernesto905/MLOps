from zenml import step
import logging
import pandas as pd 

from ..config import ModelNameConfig
from typing_extensions import Annotated

import mlflow
from zenml.client import Client
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset, Dataset

experiment_tracker = Client().active_stack.experiment_tracker


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
   

@step(experiment_tracker=experiment_tracker.name)
def train_distilbert(train_dataset: Dataset, eval_dataset: Dataset) -> Annotated[str, "model_path"]:
    """
    trains model based on ingested data

    Args:
        train_dataset: Ingested training data 
        eval_dataset: Ingested evaluation data 
    Returns: 
        Trainer: hugging face trainer for our model
    """

    try:
        model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

        # Define training arguments
        training_args = TrainingArguments(
            num_train_epochs=3,
            output_dir='./results',
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_steps=10,
            evaluation_strategy='epoch'
        )

        # Create the Trainer with compute_metrics
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics
        )

        trainer.train()
        trainer.save_model()

        return training_args.output_dir
    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        raise e

