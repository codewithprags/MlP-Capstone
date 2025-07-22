import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score, 
                           confusion_matrix, accuracy_score, precision_score, recall_score, f1_score)
import MLP as mlp
import testing as test



# Load dataset
data = pd.read_csv(r"d:\VSCode ProjectsRepos\MlP-Capstone\data\diabetes_health_indicators.csv")

# Clean and preprocess the data
df = data.copy()

# Apply the same preprocessing as MLP for fair comparison
df = mlp.clean_data(df)
df = mlp.scale_features(df)  
df = mlp.select_features(df)  

print(f"Features selected: {list(df.columns)}")

# Fix class imbalance using the same method as MLP
df_balanced = test.fix_oversampling(df)



# Prepare features and target from balanced data
X = df_balanced.drop(columns=["Diabetes_binary"])
y = df_balanced["Diabetes_binary"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Train Random Forest model
print("\n" + "="*60)
print("TRAINING RANDOM FOREST MODEL")
print("="*60)

print("\nTraining Default Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
print("\n" + "="*60)
print("MODEL EVALUATION RESULTS")
print("="*60)

print(f"\nRandom Forest Results:")
print("-" * 30)

# Predict probabilities and classes
y_scores = rf_model.predict_proba(X_test)[:, 1]
y_pred = rf_model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ROC metrics
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Precision-Recall metrics
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_scores)
average_precision = average_precision_score(y_test, y_scores)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f} ← Key metric for diabetes detection")
print(f"F1 Score: {f1:.3f}")
print(f"AUROC: {roc_auc:.3f}")
print(f"AUPRC: {average_precision:.3f}")

# Calculate the minority class proportion for the random baseline
minor_class = y_test.mean()

# Create visualizations
plt.figure(figsize=(15, 5))

# ROC Curve
plt.subplot(1, 3, 1)
plt.plot(
    fpr,
    tpr,
    color='darkorange',
    lw=2,
    label=f'ROC curve (area = {roc_auc:.3f})'
)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc="lower right")

# Precision-Recall Curve
plt.subplot(1, 3, 2)
plt.plot(
    recall_curve,
    precision_curve,
    color='blue',
    lw=2,
    label=f'PR curve (AP = {average_precision:.3f})'
)
plt.axhline(y=minor_class, color='red', lw=2, linestyle='--', label='Random')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Random Forest')
plt.legend()

# Confusion Matrix
plt.subplot(1, 3, 3)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.savefig(r'd:\VSCode ProjectsRepos\MlP-Capstone\figures\random_forest_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance analysis
feature_importance = rf_model.feature_importances_

# Get top 10 most important features
feature_names = X.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\nTOP 10 IMPORTANT FEATURES (Random Forest):")
print("-" * 50)
for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
    print(f"{i:2d}. {row['feature']:25s} {row['importance']:.4f}")

# Plot feature importance
plt.figure(figsize=(12, 8))
top_features = importance_df.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Feature Importances - Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(r'd:\VSCode ProjectsRepos\MlP-Capstone\figures\random_forest_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nRandom Forest analysis complete!")
print(f"   Model: Default Random Forest")
print(f"   Key metrics: {recall:.1%} recall, {roc_auc:.3f} AUROC")
print(f"   Plots saved: random_forest_evaluation.png, random_forest_feature_importance.png")



