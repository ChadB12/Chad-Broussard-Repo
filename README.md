# **Customer Churn Prediction Project**

## **Overview**

This project uses machine learning to predict **customer churn** based on various attributes, using a **logistic regression model**. The project focuses on a dataset of customers, specifically analyzing features such as **tenure, monthly charges, total charges**, and other categorical factors like **gender, contract type, and internet service options**. The goal is to understand which customers are more likely to leave (churn) and provide actionable insights.

## **Project Structure**

- **app.py** - Flask application that allows users to upload a dataset and get churn predictions.
- **templates/**  
  - `index.html` - Upload page for CSV files.
  - `result.html` - Result page displaying random customer data and the churn prediction.
- **churn_model_3.pkl** - The trained logistic regression model.
- **trained_features.txt** - List of features the model was trained on, ensuring compatibility with the uploaded data.
- **uploads/** - Folder to store user-uploaded files for predictions.

## **Dataset**

The dataset, `WA_Fn-UseC_-Telco-Customer-Churn.csv`, contains customer information with features such as:
- **SeniorCitizen**
- **tenure**
- **MonthlyCharges**
- **TotalCharges**
- Categorical features like **gender**, **Contract**, **InternetService**

The target variable is **Churn**, indicating whether the customer left or stayed.

## **Steps**

1. **Data Preprocessing**:
   - Convert `TotalCharges` to numeric, handling missing values.
   - One-hot encode categorical variables.
   - Add derived features:
     - **total_services**: Total active services for each customer.
     - **tenure_per_service**: Tenure divided by the number of active services.

2. **Model Training**:
   - Train a **Logistic Regression** model to classify customers as churned or not churned.
   - Split the data into training and testing sets with an 80/20 split.

3. **Flask App**:
   - A web interface built with Flask for user interaction.
   - Users can upload their own CSV file to get predictions for their data.
   - Displays a random customer’s information and whether they are likely to churn.

## **Technologies Used**

- **Python**: Programming language for building the model and Flask app.
- **Pandas**: Data manipulation and preprocessing.
- **scikit-learn**: Machine learning library used to train the logistic regression model.
- **Flask**: Web framework to build the interactive web application.
- **HTML/CSS**: Frontend for the web application.

## **How to Run**

1. **Install required packages**:
   ```bash
   pip install -r requirements.txt
2. **Run the Flask App**
   ```bash
   python app.py
3. **Upload CSV Data**
   - CSV Data is located in files (Telco Customer Churn Dataset)

***Conclusions***
- - This project provides a functional example of a machine learning powered web application that predicts customer churn.
- - By identifying churn patterns, businesses can implement targeted retention strategies to improve customer loyalty
