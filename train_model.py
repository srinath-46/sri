import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load and merge datasets
customer = pd.read_csv("customer_profile.csv")
loan = pd.read_csv("loan_history.csv")

# Merge datasets (adjust keys as needed)
data = pd.merge(customer, loan, on="customer_id")

# Target and Features
X = data.drop(columns=['loan_status', 'customer_id'])  # loan_status: 0 = no default, 1 = default
y = data['loan_status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, "loan_default_model.pkl")

# Evaluation
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
