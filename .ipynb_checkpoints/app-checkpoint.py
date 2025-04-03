from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("C:/Users/Bhoomika NS/thyroid_disease_detection/thyroid_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from the form and convert values to float
        data = request.form.to_dict()
        input_features = [float(value) for value in data.values()]
        
        # Predict class
        prediction_class = model.predict([input_features])[0]  # Numeric class
        
        # Mapping numeric classes to disease names
        class_mapping = {
            1: "Normal",
            2: "Hyperthyroidism",
            3: "Hypothyroidism"
        }
        
        prediction_name = class_mapping.get(prediction_class, "Unknown")
        
        return render_template('result.html', prediction_class=prediction_class, prediction_name=prediction_name)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)