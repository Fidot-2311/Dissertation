#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install flask joblib')


# In[2]:


from flask import Flask, request, jsonify
import joblib
import pandas as pd
from threading import Thread

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

# Function to run the Flask app
def run_app():
    app.run(debug=True, use_reloader=False)

# Run the Flask app in a separate thread
thread = Thread(target=run_app)
thread.start()


# In[3]:


get_ipython().system('pip install requests')


# In[6]:


import requests
import json

# Define the URL of the Flask API
url = 'http://127.0.0.1:5000/predict'

# Create a sample data input for prediction
data = [{
    "TouDef3rd": 3.20,
    "TI": 9.31,
    "Clr": 3.06,
    "PasTotPrgDist": 2.77,
    "PasDead": 4.69,
    "TouDefPen": 3.40,
    "PasMedCmp": 2.56,
    "TouAtt3rd": 2.51,
    "RecProg": 2.32,
    "TouAttPen": 2.31,
    "PasTotCmp": 2.62,
    "TouMid3rd": 2.42,
    "Shots": 2.22
}]

# Convert data to JSON
data_json = json.dumps(data)

# Send the POST request
response = requests.post(url, json=data)

# Print the response
print(response.json())


# In[ ]:




