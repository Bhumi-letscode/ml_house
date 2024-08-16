from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model (Make sure you've saved the model as 'random_forest_model.pkl')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    area = float(request.form['area'])
    bhk = int(request.form['bhk'])
    bathrooms = int(request.form['bathrooms'])
    furnishing = request.form['furnishing']
    locality = request.form['locality']
    parking = int(request.form['parking'])
    status = request.form['status']
    transaction = request.form['transaction']
    type_ = request.form['type']
    per_sqft = float(request.form['per_sqft'])
    
    # Prepare data for prediction (OneHotEncode categorical variables if needed)
    # Assume here that you've already handled encoding in your model pipeline
    # Assuming input_features is not a DataFrame

    input_features = np.array([[area, bhk, bathrooms, furnishing, locality, parking, status, transaction, type_, per_sqft]])

    if not isinstance(input_features, pd.DataFrame):
       input_features = pd.DataFrame(input_features) 

    # Predict the price
    prediction = model.predict(input_features)[0]
    
 
    return render_template('index2.html', prediction_text=f' ₹{prediction:,.2f}')

if __name__ == "__main__":
    app.run(debug=True)
