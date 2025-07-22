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
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, precision_score, recall_score

    # Separate features and target variable
    df = data.copy()
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Get predictions and probabilities
    y_pred = model.predict(X)
    y_scores = model.predict_proba(X)[:, 1]
    minor_class = y.mean()  # Calculate the minority class proportion for the PR baseline
    
    # Calculate performance metrics
    accuracy = accuracy_score(y, y_pred)
    precision_avg = precision_score(y, y_pred, average='binary')
    recall_avg = recall_score(y, y_pred, average='binary')
    f1_score = 2*(precision_avg*recall_avg)/(precision_avg+recall_avg)

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y, y_scores)
    roc_auc = auc(fpr, tpr)

    # Compute Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y, y_scores)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot ROC curve
    ax1.plot(fpr, tpr, color='blue', label=f'ROC (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random')
    ax1.set_title('ROC Curve - MLP Classifier')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Plot Precision-Recall curve
    ax2.plot(recall, precision, color='green', label=f'PR Curve')
    ax2.axhline(y=minor_class, color='red', lw=2, linestyle='--', label=f'Random (y={minor_class:.3f})')
    ax2.set_title('Precision-Recall Curve - MLP Classifier')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)

    # Add metrics text box in the middle-right area
    metrics_text = f"""Model Performance:
Accuracy: {accuracy:.3f}
Precision: {precision_avg:.3f}
Recall: {recall_avg:.3f}
F1-Score: {f1_score:.3f}
AUC: {roc_auc:.3f}"""
    
    # Position the text box to overlay in the top right corner
    fig.text(0.85, 0.85, metrics_text, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9, edgecolor='navy'),
             fontsize=10, ha='center', va='top',
             transform=fig.transFigure, zorder=10)

    plt.tight_layout()
    
    # Don't show immediately - let caller decide when to show/save
    # plt.show()
    return plt.gcf()  # Return the figure object


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

