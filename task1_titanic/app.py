from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model and df
# Get the directory where app.py is located
base_path = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the files
model_path = os.path.join(base_path, 'model.pkl')
df_path = os.path.join(base_path, 'df.pkl')

# Load model and df using the full paths
model = pickle.load(open(model_path, 'rb'))
df = pickle.load(open(df_path, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        pclass      = int(data['pclass'])
        sex         = int(data['sex'])          # 0=male, 1=female
        age         = int(data['age'])
        fare        = float(data['fare'])
        embarked    = int(data['embarked'])     # 0=C, 1=Q, 2=S
        family_size = int(data['family_size'])

        features = pd.DataFrame([[pclass, sex, age, fare, embarked, family_size]],
                                columns=['Pclass','Sex','Age','Fare','Embarked','Family_size'])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        return jsonify({
            'survived': int(prediction),
            'survival_probability': round(float(probability[1]) * 100, 2),
            'death_probability': round(float(probability[0]) * 100, 2),
            'message': 'Would Survive! 🎉' if prediction == 1 else 'Would Not Survive 💀'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/stats', methods=['GET'])
def stats():
    total = len(df)
    survived = int(df['Survived'].sum())
    return jsonify({
        'total_passengers': total,
        'survived': survived,
        'not_survived': total - survived,
        'survival_rate': round(survived / total * 100, 1)
    })

if __name__ == '__main__':
    app.run(debug=True)