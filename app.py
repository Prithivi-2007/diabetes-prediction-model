from flask import Flask, render_template, request
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# ============================================================
# LOAD DATASET
# ============================================================

data = pd.read_csv("diabetes.csv")

# Features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# ============================================================
# SPLIT DATASET
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# TRAIN DECISION TREE MODEL
# ============================================================

model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

model.fit(X_train, y_train)

# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("template.html")


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    pregnancies = float(request.form["pregnancies"])
    glucose = float(request.form["glucose"])
    blood_pressure = float(request.form["blood_pressure"])
    skin_thickness = float(request.form["skin_thickness"])
    insulin = float(request.form["insulin"])
    bmi = float(request.form["bmi"])
    diabetes_pedigree = float(request.form["diabetes_pedigree"])
    age = float(request.form["age"])

    # Create patient DataFrame
    patient = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]],
        columns=X.columns
    )

    # Make prediction
    prediction = model.predict(patient)[0]

    if prediction == 1:
        result = "Patient may have diabetes."
    else:
        result = "Patient is predicted as non-diabetic."

    return render_template(
        "template.html",
        prediction=result
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )