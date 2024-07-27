from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained XGBoost model
model = joblib.load('best_xgboost_model.pkl')

# Create a Flask app
app = Flask(__name__)

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Convert data to DataFrame
        df = pd.DataFrame(data)

        # Define the top 13 features
        top_13_features = [
            'TouDef3rd', 'TI', 'Clr', 'PasTotPrgDist', 'PasDead',
            'TouDefPen', 'PasMedCmp', 'TouAtt3rd', 'RecProg',
            'TouAttPen', 'PasTotCmp', 'TouMid3rd', 'Shots'
        ]

        # Ensure only the top 13 features are used
        X = df[top_13_features]

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
