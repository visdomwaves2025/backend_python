from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("crop_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = [float(request.form[key]) for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        prediction = model.predict([values])[0]
        return render_template('result.html', crop=prediction)
    except Exception as e:
        print("Error:", e)
        return render_template('result.html', crop="Error occurred! 🔁 Try Again")

if __name__ == "__main__":
    app.run(debug=True)
