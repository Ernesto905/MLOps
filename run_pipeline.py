from pipelines.training_pipeline import training_pipeline


if __name__ == "__main__":
    # Run the training pipeline
    training_pipeline(data_path="./data/olist_customers_dataset.csv")
