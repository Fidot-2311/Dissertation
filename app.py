from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Load the trained XGBoost model
model = joblib.load('XGBoost_best_model.pkl')

# Create a Flask app
app = Flask(__name__)

# Define the root route to serve the HTML file
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.json

        # Convert data to DataFrame
        df = pd.DataFrame(data)

        # Define the top n features
        top_n_features = [
    'TouDef3rd',
    'TI',
    'TouDefPen',
    'PasTotPrgDist',
    'Clr',
    'PasDead',
    'PasMedCmp',
    'TouAtt3rd',
    'TouMid3rd',
    'Shots',
    'TouAttPen',
    'RecProg',
    'PasTotCmp',
    'Pas3rd',
    'CarMis',
    'Touches',
    'SCA',
    'PasFK',
    'PasLonAtt'
]

        # Ensure only the top n features are used
        X = df[top_n_features]

        # Make predictions
        predictions = model.predict(X)
        predictions_proba = model.predict_proba(X).tolist()

        # Return the predictions as a JSON response
        return jsonify({
            'predictions': predictions.tolist(),
            'predictions_proba': predictions_proba
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
