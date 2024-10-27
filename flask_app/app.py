from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import os

# Create Flask app instance
app = Flask(__name__)

# Folder to store uploaded files
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the trained model
model = joblib.load(r'C:\Users\chadb\Documents\IOD Labs\churn_model_3.pkl')

# Load the feature names used during model training
with open('trained_features.txt', 'r') as f:
    trained_features = f.read().splitlines()

# Preprocessing function to align uploaded data with the model
def preprocess_data(df):
    """Preprocess uploaded data to align with the model's trained features."""
    # Handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Calculate 'total_services' and 'tenure_per_service'
    service_features = [
        'PhoneService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df['total_services'] = df[service_features].apply(lambda x: sum(x == 'Yes'), axis=1)
    df['total_services'].replace(0, pd.NA, inplace=True)
    df['tenure_per_service'] = df['tenure'] / df['total_services']
    df['tenure_per_service'].fillna(0, inplace=True)

    # One-hot encode the categorical columns
    categorical_columns = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    df_encoded = pd.get_dummies(df, columns=categorical_columns)

    # Align uploaded data with trained model features
    for feature in trained_features:
        if feature not in df_encoded.columns:
            df_encoded[feature] = 0  # Add missing features as 0

    return df_encoded[trained_features]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']

        # Validate file selection
        if file.filename == '':
            return "No selected file"

        if file and file.filename.endswith('.csv'):
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Load the uploaded file into a DataFrame
            df = pd.read_csv(file_path)

            # Preprocess the data
            df_preprocessed = preprocess_data(df)

            # Make predictions
            predictions = model.predict(df_preprocessed)

            # Randomly select a customer to display
            random_customer = df.sample(n=1).to_dict(orient='records')[0]
            churn_prediction = "Churn" if predictions[0] == 1 else "No Churn"

            # Render result page with customer details and prediction
            return render_template(
                'result.html',
                customer=random_customer,
                prediction_text=churn_prediction
            )

        else:
            return "Please upload a valid CSV file."
    return render_template('index.html')

@app.route('/result')
def results():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
