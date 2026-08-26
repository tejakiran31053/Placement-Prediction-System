import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def run_logistic_regression(csv_file_path=None, target_column=None):
    # 1. Load or Generate Dataset
    if csv_file_path and target_column:
        print(f"Loading data from {csv_file_path}...")
        df = pd.read_csv(csv_file_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        print("Using synthetic dataset...")
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=1000,
            n_features=5,
            n_informative=3,
            n_redundant=0,
            random_state=42
        )

    # 2. Split into Train and Test Sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Feature Scaling (Crucial for Logistic Regression convergence)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train the Model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # 5. Evaluate the Model
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 40)
    print(f"Accuracy: {acc * 100:.2f}%")
    print("=" * 40)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # 6. Save Model and Scaler for production/Flask integration
    joblib.dump(model, "logistic_regression_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("\nModel and Scaler saved as .pkl files.")

    return model, scaler

if __name__ == "__main__":
    run_logistic_regression()