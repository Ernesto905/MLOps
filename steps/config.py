from zenml.steps import BaseParameters


class ModelNameConfig(BaseParameters):
    """Mode configs"""
    model_name: str = "LinearRegression" 


