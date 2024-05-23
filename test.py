import pandas as pd 
import numpy as np 

data = pd.read_csv("./data/olist_customers_dataset.csv")
for col in data.columns:
    print(col)
