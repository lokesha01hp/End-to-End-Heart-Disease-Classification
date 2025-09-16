# CardioSense-An-ML-Approach-to-Heart-Disease-Classification (Using Sk-learn workflow)

- This project uses machine learning to classify whether a patient has heart disease, based on clinical features like age, sex, chest pain type, and max heart rate.  
- Built with Scikit-learn, it follows a full ML workflow: data cleaning, exploratory analysis, model training, hyperparameter tuning, and evaluation.

---

## Dataset

- **Source:** [heart-disease.csv](https://www.kaggle.com/datasets/formyuse/heart-disease-csv) 
- **Features:** 13 clinical attributes
- **Target:** `0 = No Heart Disease`, `1 = Heart Disease`

---

## Exploratory Data Analysis (EDA)

- Frequency plots (target, sex, chest pain types)
- Scatter plots for age vs. max heart rate
- Correlation heatmap to identify key predictors

---

## Models Used

- **Logistic Regression**
- **K-Nearest Neighbours (KNN)**
- **Random Forest Classifier**

---

## Evaluation Metrics

Evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve (if Scikit-learn version ≥ 1.2)

We also employed **5-fold cross-validation** to ensure robustness.

---

## Hyperparameter Tuning

- **RandomizedSearchCV** and **GridSearchCV** used for:
  - Logistic Regression (`C`, `solver`)
  - Random Forest (`n_estimators`, `max_depth`, etc.)

---

## Feature Importance

After training, you extracted and visualised feature coefficients from the Logistic Regression model to show which features had the most impact.

---

## Saving and Making Predictions with the Trained Model

After training and saving the best model using `joblib`, the notebook demonstrates how to:
- Load the saved model with `load()`
- Create new patient input data as a `pandas.DataFrame`
- Predict heart disease risk using `.predict()`
- Interpret results with a simple evaluation function
  
## Project Status
- Completed end-to-end ML workflow
- Next step: deploy model with Flask

