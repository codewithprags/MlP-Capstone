import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo


"""This script fetches the diabetes dataset from the UCI Machine Learning Repository"""

  
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
  
# data (as pandas dataframes) 
X = cdc_diabetes_health_indicators.data.features 
y = cdc_diabetes_health_indicators.data.targets 
  

#save as one combined csv
combined_df = pd.concat([X, y], axis=1)
combined_df.to_csv('diabetes_health_indicators.csv', index=False)






