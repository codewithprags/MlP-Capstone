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

def train_mlp(data, target_column='Diabetes_binary', test_size=0.2, random_state=42, hidden_layers=(64, 32), max_iter=100):
    # Split the data into features and target variable
    df = data.copy()
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create and train the MLP classifier
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers,
                        activation='relu',
                        solver='adam',
                        max_iter=max_iter,
                        alpha=0.001,
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


def visualize_results(model, data, target_column='Diabetes_binary'):
    # Visualize the results with AUROC and PR curves
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

    # Separate features and target variable
    df = data.copy()
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Get predicted probabilities
    y_scores = model.predict_proba(X)[:, 1]

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y, y_scores)
    roc_auc = auc(fpr, tpr)

    # Compute Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y, y_scores)

    # Plot ROC curve
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

    # Plot Precision-Recall curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='green')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.tight_layout()
    plt.show()


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

