import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score)
from sklearn.model_selection import train_test_split
import MLP as mlp
import testing as test

# Load dataset
data = pd.read_csv(r"d:\VSCode ProjectsRepos\MlP-Capstone\data\diabetes_health_indicators.csv")

# Clean and preprocess the data
df = data.copy()


df = mlp.clean_data(df)
df = mlp.scale_features(df)  
df = mlp.select_features(df)

df_balanced = test.fix_oversampling(df)



# Prepare features and target from balanced data
X = df_balanced.drop(columns=["Diabetes_binary"])
y = df_balanced["Diabetes_binary"]

# Calculate the minority class proportion for the PR baseline
minority_class = y.mean()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a logistic regression model
log_reg = LogisticRegression(C=0.001, random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)

# Predict probabilities
y_scores = log_reg.predict_proba(X_test)[:, 1]

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Compute Precision-Recall curve and PR area
precision, recall, _ = precision_recall_curve(y_test, y_scores)
average_precision = average_precision_score(y_test, y_scores)

print(f"ROC AUC: {roc_auc:.2f}")
print(f"Accuracy: {accuracy_score(y_test, log_reg.predict(X_test)):.2f}")
print(f"Precision: {precision.mean():.2f}")
print(f"Recall: {recall.mean():.2f}")
print(f"F1 Score: {2 * (precision.mean() * recall.mean()) / (precision.mean() + recall.mean()):.2f}")


# Plot ROC curve and Precision-Recall curve as subplots on the same figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot ROC curve
axes[0].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve-Logistic Regression')
axes[0].legend(loc='lower right')

# Plot Precision-Recall curve
axes[1].plot(recall, precision, color='purple', lw=2, label='Precision-Recall curve (AP = %0.2f)' % average_precision)
axes[1].plot([0, 1], [minority_class, minority_class], color='navy', lw=2, linestyle='--', label='Random Guess')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve-Logistic Regression')
axes[1].legend(loc='lower left')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

#save the figure
plt.savefig(r'd:\VSCode ProjectsRepos\MlP-Capstone\figures\logistic_regression_evaluation.png', dpi=300, bbox_inches='tight')
