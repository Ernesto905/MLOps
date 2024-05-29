# Machine Learning Pipeline Orchestration with ZenML (Proof of Concept)

## Introduction
The repository documents the development of an automated, end-to-end machine learning pipeline. Using ZenML as the orchestration layer, the objective is to create cloud-independent, production ready, infrastructure for a variety of model use cases such as sentiment analysis, prediction with linear regression, and perhaps some non-supervised workloads.  

## Note
This README serves as a dynamic document, subject to updates as the project evolves and more insights on ZenML's capabilities and its interoperability with AWS/GCP/Mlflow are gathered. As time goes on, the tools employed are subject to substitution or removal, if too many issues arise given ZenML as the backbone. More documentation will be included as the project grows. 

## Project Overview
This project aims to demonstrate the effective use of several tools for orchestrating a machine learning workflow. ZenML serves as the central component, coordinating between the following elements:

### MLFlow
Experiment tracking and model versioning.

### AWS
Sagemaker for model training and deployment of model endpoints for inference. 
S3 for artifact storage 

## Running the pipeline 
Create a data directory at the root of the project and import the presently working  into data directory of the root of the project. 

Import the datasets into the data directory at the root of the repo. 
For Sentiment Analysis: [Twitter sentiment dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
For Linear Regressions: [brazilian-ecommerce public dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

### For functionality with MLflow

Integrate MLFlow into the zenml stack 
```Bash
zenml integration install mlflow -y
```

Build your stack 
```Bash
zenml experiment-tracker register <tracker_name> --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register <stack_name> -a default -o default -d mlflow -e <tracker_name> --set
```

(OPTIONAL) Verify your stack is up and running
```Bash
zenml stack list
```


Finally, to run the pipeline, specify the model you would like to use
```Bash
python run_pipeline.py --sentiment_analysis
```
or 
```Bash
python run_pipeline.py --linear_regression
```


