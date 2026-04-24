# Therapeutic Effectiveness Prediction

Predicting the improvement score of patients undergoing Ayurvedic psychological treatment using supervised machine learning.

## Problem Statement

Given a dataset of patients with demographic, psychological, lifestyle, and Ayurvedic treatment attributes, the goal is to predict each patient's **improvement score** — a continuous measure of therapeutic effectiveness.

This was part of a competitive ML challenge evaluated on **RMSE** (Root Mean Squared Error).

## Dataset Features

- **Demographic**: Age, gender
- **Mental health indicators**: Stress, anxiety, depression levels
- **Lifestyle factors**: Sleep duration, exercise frequency, diet type, caffeine intake, screen time
- **Maslow hierarchy scores**: Physiological, safety, social, esteem, self-actualization
- **Ayurvedic treatment details**: Prakriti, vikriti, herbal prescriptions, dietary recommendations, treatment duration

## Approach

### 1. Exploratory Data Analysis
- Analyzed distributions of all features
- Identified missing values and outliers
- Explored correlations between lifestyle factors and improvement score

### 2. Feature Engineering
- Constructed composite indices from Maslow hierarchy scores
- Derived interaction features between lifestyle and treatment variables
- Encoded categorical Ayurvedic attributes

### 3. Feature Selection
- Applied correlation analysis to remove redundant features
- Used model-based feature importance (tree-based estimators)
- Recursive feature elimination for final feature set

### 4. Model Development
- Trained multiple regression models (Random Forest, Gradient Boosting, etc.)
- Used cross-validation for robust evaluation
- Compared models based on RMSE on hold-out set

## Tech Stack

- **Language**: Python
- **Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn

## Submission Format

```
patient_id,improvement_score
P001,72.454
P002,65.154
...
```

## Results

Optimized pipeline achieving competitive RMSE on the test set through targeted feature engineering and model tuning.
