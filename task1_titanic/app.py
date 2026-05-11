import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # CRITICAL: This allows your HTML file to talk to this API

# ── MODEL LOADING ──
# Using absolute paths to prevent "FileNotFoundError" on Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_pickle(filename):
    path = os.path.join(BASE_DIR, filename)
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return None

model = load_pickle('model.pkl')
df = load_pickle('df.pkl')

# ── ROUTES ──

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server'}), 500
    
    try:
        data = request.get_json()

        # Extract features (Matching the naming convention used during training)
        # We use .get() to avoid crashing if a field is missing
        pclass      = int(data.get('pclass', 3))
        sex         = int(data.get('sex', 0))        # 0=male, 1=female
        age         = float(data.get('age', 30))
        fare        = float(data.get('fare', 32.0))
        embarked    = int(data.get('embarked', 2))   # 0=C, 1=Q, 2=S
        family_size = int(data.get('family_size', 1))

        # Create DataFrame with exact column names the model expects
        features = pd.DataFrame([[pclass, sex, age, fare, embarked, family_size]],
                                columns=['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize'])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        return jsonify({
            'survived': bool(prediction),
            'survival_probability': round(float(probability[1]), 4), # Oracle HTML likes decimals (0.85)
            'message': 'Would Survive! 🎉' if prediction == 1 else 'Would Not Survive 💀'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/stats', methods=['GET'])
def stats():
    if df is None:
        return jsonify({'error': 'Dataframe not loaded'}), 500
    
    total = len(df)
    survived = int(df['Survived'].sum())
    return jsonify({
        'total_passengers': total,
        'survived': survived,
        'not_survived': total - survived,
        'survival_rate': round(survived / total * 100, 1)
    })

if __name__ == '__main__':
    # Use environment port for Render compatibility
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)