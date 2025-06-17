import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Sample customer data
customer_data = {
    "customer_id": [1, 2, 3, 4, 5],
    "age": [30, 45, 28, 34, 50],
    "income": [50000, 60000, 30000, 45000, 70000],
    "dependents": [1, 0, 2, 1, 3],
    "education": [1, 0, 1, 1, 0],
    "marital_status": [1, 0, 0, 1, 1],
    "credit_score": [720, 650, 600, 710, 680],
}
loan_data = {
    "customer_id": [1, 2, 3, 4, 5],
    "loan_amount": [200000, 150000, 100000, 180000, 250000],
    "term": [36, 24, 12, 36, 60],
    "loan_status": [0, 1, 0, 0, 1]  # 1 = default, 0 = no default
}

# Create DataFrames
customer = pd.DataFrame(customer_data)
loan = pd.DataFrame(loan_data)

# Merge datasets
data = pd.merge(customer, loan, on="customer_id")

# Feature/target
X = data.drop(columns=["loan_status", "customer_id"])
y = data["loan_status"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "loan_default_model.pkl")

print("✅ Model trained and saved as loan_default_model.pkl")
