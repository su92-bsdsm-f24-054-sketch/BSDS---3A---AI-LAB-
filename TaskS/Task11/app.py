import os
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(APP_ROOT, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "spending_model.pkl")

app = Flask(__name__)
app.secret_key = "secret123"  # simple secret for flash messages

# Try to load the model when the app starts
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

FEATURES = ["Gender", "Age", "AnnualIncome"]

@app.route("/", methods=["GET", "POST"])
def home():
    global model
    prediction = None

    if request.method == "POST":
        gender = request.form.get("gender")
        age = request.form.get("age")
        income = request.form.get("annual_income")

        if not gender or not age or not income:
            flash("Please fill all fields correctly.", "danger")
            return redirect(url_for("home"))

        try:
            age = float(age)
            income = float(income)
        except ValueError:
            flash("Age and Annual Income must be numeric.", "danger")
            return redirect(url_for("home"))

        if model is None:
            flash("Model file not found. Please train and save the model first (Task 2).", "danger")
            return redirect(url_for("home"))

        # Build input DataFrame
        input_df = pd.DataFrame([{
            "Gender": gender,
            "Age": age,
            "AnnualIncome": income,
        }])

        # Predict
        pred = model.predict(input_df)[0]
        prediction = round(float(pred), 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)