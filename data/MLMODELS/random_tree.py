import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load_data import load_data
from src.data.preprocess import (
    split_data,
    identify_features,
    handle_missing_values,
    one_hot_encode_data,
    ordinal_encode_data
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def create_model():
    model = RandomForestClassifier(
        n_estimators=100,
        max_features='sqrt',
        random_state=42,
        oob_score=True,
    )
    return model


def train_model(model, x_train, y_train):
    model.fit(x_train, y_train)
    print("\nRandom Forest trained successfully!")
    return model


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return y_pred


def main():
    df = load_data()
    print("\nOriginal Dataset Shape:", df.shape)

    x_train, x_test, y_train, y_test = split_data(df)

    numerical_features, categorical_features = identify_features(x_train)

    one_hot_features = ["Gender", "City", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs"]
    ordinal_features = ["CollegeTier", "CGPA_Tier"]

    x_train, x_test, imputer = handle_missing_values(x_train, x_test, numerical_features)
    x_train, x_test, one_hot_encoder = one_hot_encode_data(x_train, x_test, one_hot_features)
    x_train, x_test, ordinal_encoder = ordinal_encode_data(x_train, x_test, ordinal_features)

    model = create_model()
    model = train_model(model, x_train, y_train)
    evaluate_model(model, x_test, y_test)


if __name__ == "__main__":
    main()