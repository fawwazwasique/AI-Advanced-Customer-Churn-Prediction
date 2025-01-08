import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Load dataset
@st.cache
def load_data():
    data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return data

# Preprocess data
def preprocess_data(data):
    # Create tenure interval
    def group_tenure(tenure):
        if 0 <= tenure <= 6:
            return '0-6 Month'
        elif 6 < tenure <= 12:
            return '6-12 Month'
        elif 12 < tenure <= 24:
            return '12-24 Month'
        elif 24 < tenure <= 36:
            return '24-36 Month'
        elif 36 < tenure <= 48:
            return '36-48 Month'
        elif 48 < tenure <= 62:
            return '48-62 Month'
        elif tenure > 62:
            return '> 62 Month'

    data['tenure_interval'] = data['tenure'].apply(group_tenure)
    data['tenure_interval'] = data['tenure_interval'].astype('category')

    # Drop customerID and tenure columns
    data.drop(['customerID', 'tenure'], axis=1, inplace=True)

    # Clean up specific columns
    columns_to_clean = ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
                        "TechSupport", "StreamingTV", "StreamingMovies"]
    for column in columns_to_clean:
        data[column] = data[column].replace({"No phone service": "No", "No internet service": "No"})
        data[column] = data[column].astype('category')

    # Drop rows with NA values
    data.dropna(inplace=True)

    return data

# Load and preprocess the dataset
data = load_data()
data = preprocess_data(data)

# Split data for training
X = data.drop("Churn", axis=1)
X = pd.get_dummies(X, drop_first=True)
y = data['Churn'].replace({"No": 0, "Yes": 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Streamlit Dashboard
st.title("Telecom Customer Churn Prediction")

# Sidebar for individual input
st.sidebar.header("Customer Data Input")
input_data = {
    "gender": st.sidebar.selectbox("Gender", ["Male", "Female"]),
    "SeniorCitizen": st.sidebar.selectbox("Senior Citizen", [0, 1]),
    "Partner": st.sidebar.selectbox("Partner", ["Yes", "No"]),
    "Dependents": st.sidebar.selectbox("Dependents", ["Yes", "No"]),
    "PhoneService": st.sidebar.selectbox("Phone Service", ["Yes", "No"]),
    "MultipleLines": st.sidebar.selectbox("Multiple Lines", ["Yes", "No"]),
    "InternetService": st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
    "OnlineSecurity": st.sidebar.selectbox("Online Security", ["Yes", "No"]),
    "OnlineBackup": st.sidebar.selectbox("Online Backup", ["Yes", "No"]),
    "DeviceProtection": st.sidebar.selectbox("Device Protection", ["Yes", "No"]),
    "TechSupport": st.sidebar.selectbox("Tech Support", ["Yes", "No"]),
    "StreamingTV": st.sidebar.selectbox("Streaming TV", ["Yes", "No"]),
    "StreamingMovies": st.sidebar.selectbox("Streaming Movies", ["Yes", "No"]),
    "Contract": st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
    "PaperlessBilling": st.sidebar.selectbox("Paperless Billing", ["Yes", "No"]),
    "PaymentMethod": st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]),
    "MonthlyCharges": st.sidebar.slider("Monthly Charges", 0.0, 200.0, 50.0),
    "TotalCharges": st.sidebar.slider("Total Charges", 0.0, 10000.0, 500.0),
    "tenure_interval": st.sidebar.selectbox("Tenure Interval", ['0-6 Month', '6-12 Month', '12-24 Month', '24-36 Month', '36-48 Month', '48-62 Month', '> 62 Month']),
}

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Preprocess input data
input_processed = pd.get_dummies(input_df, drop_first=True)
input_processed = input_processed.reindex(columns=X.columns, fill_value=0)

# Predict churn for individual input
if st.button("Predict Churn (Individual)"):
    prediction = model.predict(input_processed)
    prediction_proba = model.predict_proba(input_processed)[:, 1]
    st.write("Prediction: **Churn**" if prediction[0] == 1 else "Prediction: **No Churn**")
    st.write(f"Probability of Churn: **{prediction_proba[0]:.2f}**")

# File upload for batch predictions
st.header("Batch Churn Prediction")
uploaded_file = st.file_uploader("Upload CSV File for Batch Prediction", type=["csv"])

if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    # Preprocess the uploaded batch data
    batch_data = preprocess_data(batch_data)
    batch_processed = pd.get_dummies(batch_data, drop_first=True)
    batch_processed = batch_processed.reindex(columns=X.columns, fill_value=0)
    
    # Predict churn for the batch data
    batch_predictions = model.predict(batch_processed)
    batch_probabilities = model.predict_proba(batch_processed)[:, 1]
    
    # Combine results
    batch_data['Churn_Prediction'] = ["Churn" if pred == 1 else "No Churn" for pred in batch_predictions]
    batch_data['Churn_Probability'] = batch_probabilities
    
    # Display results
    st.write("Batch Prediction Results:")
    st.write(batch_data)
    
    # Option to download the results
    csv = batch_data.to_csv(index=False)
    st.download_button("Download Prediction Results", data=csv, file_name="batch_predictions.csv", mime="text/csv")

# Display model performance
st.header("Model Performance")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

st.write(f"Accuracy: **{accuracy:.2f}**")
st.write("Confusion Matrix:")
st.write(conf_matrix)
