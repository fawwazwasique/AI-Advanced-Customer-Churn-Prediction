# Telco Customer Churn Prediction Dashboard

This project is a **Streamlit**-based web application for predicting customer churn in a telecom dataset. The application allows users to predict churn for individual customers as well as batch predictions by uploading a CSV file.

---

## Features

1. **Individual Customer Churn Prediction**
   - Input customer data manually using the sidebar.
   - Get real-time predictions for churn status ("Churn" or "No Churn") and the probability of churn.

2. **Batch Prediction via File Upload**
   - Upload a CSV file containing customer data.
   - Predict churn for all records in the file.
   - Download the prediction results as a CSV file.

3. **Model Performance Metrics**
   - Displays the model's accuracy and confusion matrix on the test dataset.

---

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/telco-churn-dashboard.git
   cd telco-churn-dashboard
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Access the app in your web browser at `http://localhost:8501`.

---

## Data Requirements

The application expects customer data in the following format:

### Columns:
- `gender`: Gender of the customer (e.g., Male, Female).
- `SeniorCitizen`: Whether the customer is a senior citizen (0 = No, 1 = Yes).
- `Partner`: Whether the customer has a partner (Yes/No).
- `Dependents`: Whether the customer has dependents (Yes/No).
- `PhoneService`: Whether the customer has phone service (Yes/No).
- `MultipleLines`: Whether the customer has multiple lines (Yes/No).
- `InternetService`: Type of internet service (DSL, Fiber optic, No).
- `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`: Availability of these services (Yes/No).
- `Contract`: Type of contract (Month-to-month, One year, Two year).
- `PaperlessBilling`: Whether the customer uses paperless billing (Yes/No).
- `PaymentMethod`: Payment method (e.g., Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)).
- `MonthlyCharges`: Monthly charges for the customer.
- `TotalCharges`: Total charges for the customer.
- `tenure_interval`: Customer tenure in categories (e.g., '0-6 Month', '6-12 Month', etc.).

Ensure the uploaded file matches this structure for batch predictions.

---

## File Upload for Batch Predictions

1. Prepare a CSV file with the same structure as described above.
2. Upload the file using the **"Batch Churn Prediction"** section in the app.
3. View predictions directly in the app or download the results as a CSV file.

---

## Model Details

- **Algorithm**: Logistic Regression
- **Training**:
  - Dataset: [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
  - Preprocessing includes:
    - Feature engineering (e.g., grouping `tenure` into intervals).
    - Encoding categorical variables using one-hot encoding.
    - Cleaning specific columns (e.g., replacing "No internet service" with "No").

---

## Key Python Libraries

- **Streamlit**: For building the interactive dashboard.
- **pandas**: For data manipulation and preprocessing.
- **scikit-learn**: For machine learning model training and evaluation.

---

## File Structure

- `app.py`: Main application code.
- `requirements.txt`: Python dependencies.
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: Sample dataset (optional).

---

## Future Enhancements

- Add advanced visualizations for predictions and metrics.
- Integrate other machine learning models for comparison.
- Enhance error handling for invalid file uploads.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn).
- Built using [Streamlit](https://streamlit.io/).

