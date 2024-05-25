# Machine Learning Pipeline Orchestration with ZenML (Proof of Concept)

## Introduction
The repository documents the development of an automated, end-to-end machine learning pipeline. The objective is to create a cloud-independent, production ready, infrastructure for sentiment analysis, utilizing ZenML as the orchestration layer.

## Note
This README serves as a dynamic document, subject to updates as the project evolves and more insights on ZenML's capabilities and its interaction with AWS Datazone are gathered. As time goes on, the tools employed are subject to substitution or removal, if too many issues arise given ZenML as the backbone. More documentation will be included as the project grows.  

## Project Overview
This project aims to demonstrate the effective use of several tools for orchestrating a machine learning workflow. ZenML serves as the central component, coordinating between the following elements:

### MLFlow
Integrated within ZenML, MLFlow provides experiment tracking and model versioning.

### Airflow
Airflow schedules and automates pipeline execution, creating DAGs for consistent operation.

### SageMaker
Used for compute resources for model training and hosting the inference service.

### AWS Datazone
Datazone organizes and governs the data catalog, ensuring datasets are ready for consumption within the pipeline.

