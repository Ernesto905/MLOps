from pipelines.training_pipeline import training_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
import click 

from zenml.client import Client

@click.command()
@click.option('--data', help='What data will you use')


if __name__ == "__main__":
    """Choose a pipeline in accordance with your desired functionality"""


    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri \"{get_tracking_uri()}\"\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )
    print(Client().active_stack.experiment_tracker.get_tracking_uri())

    # Run the training pipeline
    # training_pipeline(data_path="./data/olist_customers_dataset.csv")
    sentiment_pipeline(data_path="./data/twitter_training.csv")
