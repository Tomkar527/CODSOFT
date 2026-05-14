from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Label map (reverse of training encoding)
label_map = {0: "Iris-Versicolor", 1: "Iris-Virginica", 2: "Iris-Setosa"}
emoji_map = {0: "🌸", 1: "🌺", 2: "🌼"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = pd.DataFrame([{
            'sepal_length': float(data['sepal_length']),
            'sepal_width':  float(data['sepal_width']),
            'petal_length': float(data['petal_length']),
            'petal_width':  float(data['petal_width']),
        }])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        result = {
            "prediction": label_map[prediction],
            "emoji": emoji_map[prediction],
            "confidence": round(float(max(probabilities)) * 100, 2),
            "probabilities": {
                label_map[i]: round(float(p) * 100, 2)
                for i, p in enumerate(probabilities)
            }
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
