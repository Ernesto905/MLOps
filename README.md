# Machine Learning Pipeline Orchestration with ZenML

## Introduction
This repository showcases an automated, end-to-end machine learning pipeline, from data preprocessing to model training. It uses ZenML as the orchestration layer. It provides infrastructure for training a distil-bert model on sentiment analysis, as well as a simple linear regression model. The models used are small enough to run locally on most laptops. 

## Project Overview
The following tools were used in the creation of this pipeline

### ZenML
ZenML serves as the central component, coordinating between the different pipeline stages

### MLFlow
Experiment tracking and model metric visualization

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


