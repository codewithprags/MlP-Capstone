# Multi-layer Perceptron Capstone Project
## Diabetes Prediction Using Machine Learning


### 📋 Project Overview

This capstone project focuses on predicting diabetes using Multi-layer Perceptron (MLP) neural networks and comparing performance with other machine learning algorithms. The project is part of the Clinical Bioinformatics program at Humber Polytechnic.

**Key Objectives:**
- Develop and optimize MLP models for diabetes prediction
- Address class imbalance challenges in medical datasets
- Compare MLP performance with Random Forest and Logistic Regression

### 🏥 Clinical Significance

Early diabetes detection is crucial for:
- **Preventing complications**: Cardiovascular disease, kidney failure, blindness
- **Reducing healthcare costs**: Early intervention is more cost-effective
- **Improving patient outcomes**: Better quality of life with early treatment
- **Population health screening**: Identifying at-risk individuals

### 📊 Dataset

**Source**: UCI ML Repository - Diabetes Health Indicators Dataset  
**Size**: 253,680 survey responses from CDC BRFSS 2015  
**Features**: 21 health indicators including:
- Demographics (Age, Sex, Education, Income)
- Health metrics (BMI, Blood Pressure, Cholesterol)
- Lifestyle factors (Smoking, Physical Activity, Diet)
- Medical history (Heart Disease, Stroke, Mental Health)

**Target Variable**: Binary diabetes diagnosis (0: No Diabetes, 1: Diabetes)

### 🔬 Methodology

#### Data Preprocessing
1. **Data Cleaning**: Remove duplicates and handle missing values
2. **Feature Scaling**: StandardScaler normalization
3. **Feature Selection**: SelectKBest with ANOVA F-test (top 15 features)
4. **Class Balancing**: SMOTE/resample oversampling to address imbalance

#### Model Architecture
- **MLP Configuration**: Multiple architectures tested
  - Model 1: (64, 32) hidden layers
  - Model 2: (64, 32) hidden layers (no feature selection)
  - Model 3: (128, 64, 32) hidden layers
- **Activation**: ReLU
- **Solver**: Adam optimizer
- **Regularization**: L2 (alpha=0.001), Early stopping


