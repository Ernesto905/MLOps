from pipelines.training_pipeline import training_pipeline
from pipelines.sentiment_pipeline import sentiment_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
import click 

from zenml.client import Client

@click.command(
    help="""
    ZenML End-To-End ML Pipeline Proof of Concept v0.0.1.

    Currently, we support two pipelines options.

    Examples: 
        # Run the linear regression training pipeline 
        python run_pipeline.py --linear_regression

        # Run the sentiment analysis training pipeline 
        python run_pipeline.py --sentiment_analysis


    """
)

@click.option(
        '--linear_regression',
        is_flag=True,
        default=False,
        help="Run linear regression model"
)

@click.option(
        '--sentiment_analysis',
        is_flag=True,
        default=False,
        help="Run linear regression model"
)

def main(linear_regression: bool = False, sentiment_analysis: bool = False):
    """Choose a pipeline in accordance with your desired functionality"""


    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri \"{get_tracking_uri()}\"\n"
        "To inspect your experiment runs within the mlflow UI.\n"
    )


    # Run the training pipeline
    if linear_regression:
        training_pipeline(data_path="./data/olist_customers_dataset.csv")
    elif sentiment_analysis:
        sentiment_pipeline(data_path="./data/twitter_training.csv")
    else:
        print("Please specify a model training workflow")





if __name__ == "__main__":
    main()
