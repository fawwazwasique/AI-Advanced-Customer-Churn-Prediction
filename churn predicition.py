import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Read the telecom dataset
telecom_dataframe = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Print the structure of the dataframe
print(telecom_dataframe.info())

# Check for NA values
print(telecom_dataframe.isna().sum().any())

# Create a new column "tenure_interval" from the tenure column
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

telecom_dataframe['tenure_interval'] = telecom_dataframe['tenure'].apply(group_tenure)
telecom_dataframe['tenure_interval'] = telecom_dataframe['tenure_interval'].astype('category')

# Drop "customerID" and "tenure" columns
telecom_dataframe.drop(['customerID', 'tenure'], axis=1, inplace=True)

# Clean up the data for specific columns
columns_to_clean = ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
for column in columns_to_clean:
    telecom_dataframe[column] = telecom_dataframe[column].replace({"No phone service": "No", "No internet service": "No"})

# Convert character variables to categorical variables
for column in columns_to_clean:
    telecom_dataframe[column] = telecom_dataframe[column].astype('category')

# Drop rows with NA values
telecom_dataframe.dropna(inplace=True)

# Split the dataset into training and testing sets
X = telecom_dataframe.drop("Churn", axis=1)
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variables to dummy/indicator variables
y = telecom_dataframe['Churn'].replace({"No": 0, "Yes": 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test the model with the test dataset
predictions = model.predict(X_test)
predicted_probabilities = model.predict_proba(X_test)[:, 1]

# Calculate the misclassification rate and accuracy rate
misclassification_error = np.mean(predictions != y_test)
accuracy_rate = accuracy_score(y_test, predictions)

print(f"Misclassification Error: {misclassification_error}")
print(f"Accuracy Rate: {accuracy_rate}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Combine actual results with predicted results
results = pd.DataFrame({"Predicted": predictions, "Actual": y_test.values})
print(results)
