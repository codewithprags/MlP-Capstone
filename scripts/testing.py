#import required libraries for MLP
import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
import MLP as mlp
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample


diabetes_data = pd.read_csv(r"d:\VSCode ProjectsRepos\MlP-Capstone\data\diabetes_health_indicators.csv")

#print(f"Data loaded successfully! Shape: {diabetes_data.shape}")

#cleaning the data

#data= diabetes_data.copy()
#clean_data = mlp.clean_data(data)
#clean_data = mlp.scale_features(clean_data)
#trained_model = mlp.train_mlp(clean_data)




def train_mlp_balanced(data, target_column='Diabetes_binary', test_size=0.2, random_state=42, 
                                hidden_layers=(64, 32), max_iter=100):
    
    # Split the data
    df = data.copy()
    X = df.drop(columns=[target_column])
    y = df[target_column]

    #use the balanced data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state, stratify=y)
    
    #fit the MLP model
    mlp_model = MLPClassifier(hidden_layer_sizes=hidden_layers, 
                              max_iter=max_iter, 
                              activation='relu',
                              solver='adam',
                              random_state=random_state)
    
    # Fit the model on the training data
    mlp_model.fit(X_train, y_train)

    # Evaluate the model on the test data
    y_pred = mlp_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    return mlp_model




def check_balance_data(data, target_column='Diabetes_binary'):
    # Check the balance of the target variable
    class_counts = data[target_column].value_counts()
    print(f"Class distribution in the dataset:\n{class_counts}")
    
    # Plot the class distribution
    sns.countplot(x=target_column, data=data)
    plt.title('Class Distribution')
    plt.xlabel(target_column)
    plt.ylabel('Count')
    plt.show()


#check_balance_data(clean_data)


def fix_oversampling(data, target_column='Diabetes_binary'):
    from sklearn.utils import resample
    
    # Separate majority and minority classes
    majority_class = data[data[target_column] == 0]
    minority_class = data[data[target_column] == 1]
    
    # Upsample minority class to match majority class
    minority_upsampled = resample(minority_class,
                                 replace=True,  # sample with replacement
                                 n_samples=len(majority_class),  # match majority class size
                                 random_state=42)
    
    # Combine majority class with upsampled minority class
    balanced_data = pd.concat([majority_class, minority_upsampled])
    
    return balanced_data


#clean_data = fix_oversampling(clean_data)
#trained_model2 = train_mlp_balanced(clean_data)
#print("Model training completed successfully!")
# Visualize the results
#mlp.visualize_results(trained_model2, clean_data)





