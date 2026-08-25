# ============================================================
# DIABETES PREDICTION MODEL
# Using Pima Indians Diabetes Dataset
# ============================================================

# Step 1: Import required libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# Step 2: Load the dataset
# ============================================================

# Make sure diabetes.csv is in the same folder as this Python file
data = pd.read_csv("diabetes.csv")

# Display first 5 rows
print("First 5 rows of the dataset:")
print(data.head())

# Display dataset information
print("\nDataset Information:")
print(data.info())

# Display dataset shape
print("\nDataset Shape:")
print(data.shape)

# ============================================================
# Step 3: Check for missing values
# ============================================================

print("\nMissing Values:")
print(data.isnull().sum())

# ============================================================
# Step 4: Separate features and target
# ============================================================

# X = Input features
# y = Target variable

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

print("\nFeatures:")
print(X.columns)

print("\nTarget:")
print("0 = No Diabetes")
print("1 = Diabetes")

# ============================================================
# Step 5: Split dataset into training and testing sets
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)

# ============================================================
# Step 6: Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# Step 7: Train Logistic Regression Model
# ============================================================

logistic_model = LogisticRegression(max_iter=1000)

logistic_model.fit(
    X_train_scaled,
    y_train
)

# Make predictions
logistic_predictions = logistic_model.predict(X_test_scaled)

# ============================================================
# Step 8: Train Decision Tree Model
# ============================================================

decision_tree_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

decision_tree_model.fit(
    X_train,
    y_train
)

# Make predictions
decision_tree_predictions = decision_tree_model.predict(X_test)

# ============================================================
# Step 9: Train Random Forest Model
# ============================================================

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(
    X_train,
    y_train
)

# Make predictions
random_forest_predictions = random_forest_model.predict(X_test)

# ============================================================
# Step 10: Function to evaluate models
# ============================================================

def evaluate_model(model_name, y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print("Accuracy :", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall   :", round(recall * 100, 2), "%")
    print("F1 Score :", round(f1 * 100, 2), "%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["No Diabetes", "Diabetes"],
        zero_division=0
    ))

# ============================================================
# Step 11: Evaluate all models
# ============================================================

evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_predictions
)

evaluate_model(
    "Decision Tree",
    y_test,
    decision_tree_predictions
)

evaluate_model(
    "Random Forest",
    y_test,
    random_forest_predictions
)

# ============================================================
# Step 12: Compare model accuracy
# ============================================================

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

decision_tree_accuracy = accuracy_score(
    y_test,
    decision_tree_predictions
)

random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_predictions
)

print("\n" + "=" * 60)
print("MODEL ACCURACY COMPARISON")
print("=" * 60)

print(
    "Logistic Regression:",
    round(logistic_accuracy * 100, 2),
    "%"
)

print(
    "Decision Tree:",
    round(decision_tree_accuracy * 100, 2),
    "%"
)

print(
    "Random Forest:",
    round(random_forest_accuracy * 100, 2),
    "%"
)

# ============================================================
# Step 13: Select the best model
# ============================================================

models = {
    "Logistic Regression": logistic_accuracy,
    "Decision Tree": decision_tree_accuracy,
    "Random Forest": random_forest_accuracy
}

best_model_name = max(
    models,
    key=models.get
)

print("\nBest Model:", best_model_name)
print(
    "Best Accuracy:",
    round(models[best_model_name] * 100, 2),
    "%"
)

# ============================================================
# Step 14: Predict diabetes for a new patient
# ============================================================

# Patient details:
# [Pregnancies, Glucose, BloodPressure, SkinThickness,
#  Insulin, BMI, DiabetesPedigreeFunction, Age]

new_patient = pd.DataFrame(
    [[2, 120, 70, 25, 80, 28.5, 0.35, 35]],
    columns=X.columns
)

# Scale the new patient for Logistic Regression
new_patient_scaled = scaler.transform(new_patient)

# Logistic Regression prediction
logistic_result = logistic_model.predict(
    new_patient_scaled
)

# Random Forest prediction
random_forest_result = random_forest_model.predict(
    new_patient
)

print("\n" + "=" * 60)
print("NEW PATIENT PREDICTION")
print("=" * 60)

if logistic_result[0] == 1:
    print("Logistic Regression: Patient may have diabetes.")
else:
    print("Logistic Regression: Patient is predicted as non-diabetic.")

if random_forest_result[0] == 1:
    print("Random Forest: Patient may have diabetes.")
else:
    print("Random Forest: Patient is predicted as non-diabetic.")