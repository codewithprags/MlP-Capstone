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

#data = pd.read_csv('diabetes_health_indicators.csv')


def clean_data(data):

    #copy the data to avoid modifying the original dataframe
    df = data.copy()
    # Remove duplicates
    df = df.drop_duplicates()
    # Fill remove rows with missing values
    df = df.dropna()

    return df

def cor_matrix(data):
    # Calculate the correlation matrix as heatmap
    corr = data.corr()

    #plot the heatmap 
   
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix Heatmap')
    plt.show()
    
    return corr


def scale_features(data, method='standard', target_column='Diabetes_binary'):
    # Scale features using the specified method for numerical features
    # Exclude target column from scaling
    df = data.copy()
    
    # Separate features and target
    if target_column in df.columns:
        target = df[target_column]
        features = df.drop(columns=[target_column])
    else:
        features = df
        target = None
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Method must be 'standard' or 'minmax'")

    # Fit and transform only the features
    scaled_features = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    
    # Add target column back if it exists
    if target is not None:
        scaled_df[target_column] = target.values
    
    return scaled_df

def select_features(data, num_features=15, target_column='Diabetes_binary'):
    # Select the top features based on correlation with the target column
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Use SelectKBest to select the top features
    selector = SelectKBest(score_func=f_classif, k=num_features)
    X_new = selector.fit_transform(X, y)

    # Get the selected feature names
    selected_features = X.columns[selector.get_support()]

    #return the transformed data and selected features ensure target colum is included
    X_new = pd.DataFrame(X_new, columns=selected_features)
    new_df = pd.DataFrame(X_new, columns=selected_features)
    new_df[target_column] = y

    return new_df

def train_mlp(data, target_column='Diabetes_binary', test_size=0.2, random_state=42):
    # Split the data into features and target variable
    df = data.copy()
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create and train the MLP classifier
    mlp = MLPClassifier(hidden_layer_sizes=(64,32),
                        activation='relu',
                        solver='adam',
                        max_iter=100,
                        batch_size=32,
                        learning_rate_init=0.001,
                        early_stopping=True,
                        random_state=random_state)
    
    # Fit the model on the training data
    mlp.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = mlp.predict(X_test)

    # Print classification report and confusion matrix
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    return mlp

