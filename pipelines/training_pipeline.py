from zenml import pipeline
from steps.ingest_data import ingest_data 
from steps.clean_data import clean_data 
from steps.model_train import model_train
from steps.evaluation import evaluate_model

@pipeline
def training_pipeline(data_path: str):
    df = ingest_data(data_path)
    df = clean_data(df)
    model_train(df)
    evaluate_model(df)

