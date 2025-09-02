# Adult Census Education Level Prediction

## Project Overview

This project implements a machine learning solution to predict an individual's highest educational level based on demographic and socioeconomic features from the Adult Census dataset. The project follows the complete machine learning lifecycle from data exploration and preprocessing to model training, evaluation, and optimization.

## Dataset

The project uses the Adult Census dataset (`censusData.csv`), which contains demographic information from the 1994 Census. The dataset includes 15 features with 32,561 records.

**Original Features:**
- age: Continuous numerical feature
- workclass: Categorical employment type
- fnlwgt: Continuous numerical feature (final weight)
- education: Categorical education level (target variable)
- education-num: Numerical education level encoding
- marital-status: Categorical marital status
- occupation: Categorical occupation type
- relationship: Categorical relationship status
- race: Categorical race
- sex_selfID: Categorical gender identification
- capital-gain: Continuous numerical investment income
- capital-loss: Continuous numerical investment losses
- hours-per-week: Continuous numerical work hours
- native-country: Categorical country of origin
- income_binary: Categorical income classification

## Problem Definition

**Prediction Task:** Binary classification of education level
- **Target Variable:** `education_binary` (derived from `education` column)
- **Classes:** 'College Degree' vs 'No College Degree'

**Problem Type:** Supervised learning classification problem

**Business Value:** This model can help educational companies tailor products to specific audience segments or assist financial institutions in assessing creditworthiness based on education levels.

## Data Preprocessing

### Handling Missing Values
- Numerical features: Imputed with mean values
- Categorical features: Filled with 'Missing' category

### Feature Engineering
- Created binary variables for `workclass` and `marital-status`
- Winsorized `education-num` to handle outliers (1% on both ends)

### Categorical Encoding
- One-hot encoding applied to all categorical features
- Reduced cardinality for `native-country` by grouping rare categories

### Target Variable Transformation
- Created binary classification target:
  - 'College Degree': ['Bachelors', 'Masters', 'Doctorate', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', 'Some-college']
  - 'No College Degree': All other education levels

### Feature Scaling
- Standardized all numerical features using StandardScaler

## Modeling Approach

### Algorithms Implemented
1. **Logistic Regression** with GridSearch for hyperparameter tuning
2. **Random Forest Classifier** with GridSearch for hyperparameter optimization

### Evaluation Methodology
- Train-test split: 70-30 stratified split
- 5-fold cross-validation for hyperparameter tuning
- Primary evaluation metric: AUC-ROC (accounting for class imbalance)
- Secondary metrics: Accuracy, Precision, Recall, F1-score

### Hyperparameter Tuning
- **Logistic Regression:** Regularization strength (C parameter)
- **Random Forest:** n_estimators and max_depth parameters

## Results

### Model Performance
- **Logistic Regression:** AUC = 0.777
- **Random Forest:** Performance metrics available in the notebook

### Key Findings
- The dataset shows mild class imbalance (ratio: 1.2)
- Both models achieved reasonable performance on the classification task
- Feature importance analysis revealed the most predictive features

## Project Structure

The Jupyter notebook (`DefineAndSolveMLProblem.ipynb`) contains the complete implementation:

1. **Data Loading and Exploration**
2. **Problem Definition**
3. **Data Preprocessing and Cleaning**
4. **Feature Engineering**
5. **Model Training and Evaluation**
6. **Results Analysis**

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn

## Usage

1. Ensure all dependencies are installed
2. Place the `censusData.csv` file in the `data/` directory
3. Run the Jupyter notebook sequentially
4. The notebook will output model performance metrics and visualizations

## Future Improvements

- Address class imbalance more aggressively if needed
- Experiment with additional algorithms (Gradient Boosting, SVM)
- Perform more extensive feature selection
- Implement advanced encoding techniques for categorical variables
- Deploy the model as a web service for predictions
