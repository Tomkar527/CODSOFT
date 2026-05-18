from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and data
model = pickle.load(open('model.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

@app.route('/')
def index():
    stats = {
        'tv_min': round(df['TV'].min(), 1),
        'tv_max': round(df['TV'].max(), 1),
        'radio_min': round(df['Radio'].min(), 1),
        'radio_max': round(df['Radio'].max(), 1),
        'news_min': round(df['Newspaper'].min(), 1),
        'news_max': round(df['Newspaper'].max(), 1),
        'avg_sales': round(df['Sales'].mean(), 2),
    }
    return render_template('index.html', stats=stats)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        tv = float(data['tv'])
        radio = float(data['radio'])
        newspaper = float(data['newspaper'])

        features = np.array([[tv, radio, newspaper]])
        prediction = model.predict(features)[0]

        return jsonify({
            'success': True,
            'prediction': round(float(prediction), 2),
            'inputs': {'tv': tv, 'radio': radio, 'newspaper': newspaper}
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/data-stats')
def data_stats():
    stats = df.describe().round(2).to_dict()
    return jsonify(stats)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
