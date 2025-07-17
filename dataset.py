import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo


"""This script fetches the diabetes dataset from the UCI Machine Learning Repository"""
#diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296) 
  
data (as pandas dataframes) 
X = diabetes_130_us_hospitals_for_years_1999_2008.data.features 
y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets 
  
 metadata 
print(diabetes_130_us_hospitals_for_years_1999_2008.metadata) 

#save as one combined csv
combined_df = pd.concat([X, y], axis=1)
combined_df.to_csv('diabetes_130_us_hospitals_combined.csv', index=False)






