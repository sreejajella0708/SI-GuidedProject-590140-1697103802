import numpy as np
import pandas as pd
from joblib import load
from flask import Flask, render_template, request

app = Flask(__name__)
model = load("xg_model.joblib")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST", "GET"])
def predict():
    if request.method == 'POST':
        try:
            input_features = [float(x) for x in request.form.values()]
            feature_names = ['playerId', 'Sex', 'Equipment', 'Age', 'BodyweightKg', 'BestSquatKg', 'BestBenchKg']
            data = pd.DataFrame([input_features], columns=feature_names)
            prediction = model.predict(data)
            print("Prediction:", prediction)
            text = "Estimated Deadlift for the builder is:"
            return render_template("index.html", prediction_text=text + str(prediction))
        except Exception as e:
            error_message = "An error occurred: " + str(e)
            return render_template("error.html", error_message=error_message)

if __name__ == '__main__':
    app.debug = True
    app.run()
