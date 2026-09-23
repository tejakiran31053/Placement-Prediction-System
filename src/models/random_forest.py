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
        oob_score=True
    )
    return model


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    print("\nRandom Forest Trained Successfully!")
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"OOB Score: {model.oob_score_}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return y_pred


def main():
    df = load_data()
    print("Original Dataset Shape:")
    print(df.shape)

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column="PlacementStatus",
        drop_columns=["StudentID", "Salary Package", "IsAnomaly"]
    )

    numerical_features, categorical_features = identify_features(X_train)

    one_hot_features = ["Gender", "City", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs"]
    ordinal_features = ["CollegeTier", "CGPA_Tier"]

    X_train, X_test, imputer = handle_missing_values(X_train, X_test, numerical_features)
    X_train, X_test, one_hot_encoder = one_hot_encode_data(X_train, X_test, one_hot_features)
    X_train, X_test, ordinal_encoder = ordinal_encode_data(X_train, X_test, ordinal_features)

    model = create_model()
    model = train_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
