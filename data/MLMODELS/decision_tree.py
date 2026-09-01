import matplotlib.pyplot as plt

from src.data.load_data import load_data
from src.data.preprocess import (
    split_data,
    identify_features,
    handle_missing_values,
    one_hot_encode_data,
    ordinal_encode_data
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree


def create_model():
    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=5,
        random_state=42
    )
    return model


def train_model(model, x_train, y_train):
    model.fit(x_train, y_train)

    print("\nDecision Tree trained successfully")

    return model


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return y_pred


def display_tree(model, feature_names):
    plt.figure(figsize=(25, 12))

    tree.plot_tree(
        model,
        feature_names=feature_names,
        class_names=["Not Placed", "Placed"],
        filled=True,
        rounded=True,
        fontsize=8
    )

    plt.title("ID3 Decision Tree - Placement Prediction")

    plt.show()


def main():

    df = load_data()

    print("\nOriginal Dataset Shape:")
    print(df.shape)

    x_train, x_test, y_train, y_test = split_data(df)

    print("\nTraining Data Shape:")
    print(x_train.shape)

    print("\nTesting Data Shape:")
    print(x_test.shape)

    numerical_features, categorical_features = identify_features(x_train)

    print("\nNumerical Features:")
    print(numerical_features)

    print("\nCategorical Features:")
    print(categorical_features)

    one_hot_features = [
        "Gender",
        "City",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs"
    ]

    ordinal_features = [
        "CollegeTier",
        "CGPA_Tier"
    ]

    x_train, x_test, imputer = handle_missing_values(
        x_train,
        x_test,
        numerical_features
    )

    print("\nMissing Value Handling Completed.")

    x_train, x_test, one_hot_encoder = one_hot_encode_data(
        x_train,
        x_test,
        one_hot_features
    )

    print("\nOne-Hot Encoding Completed.")

    x_train, x_test, ordinal_encoder = ordinal_encode_data(
        x_train,
        x_test,
        ordinal_features
    )

    print("\nOrdinal Encoding Completed.")

    model = create_model()

    model = train_model(
        model,
        x_train,
        y_train
    )

    evaluate_model(
        model,
        x_test,
        y_test
    )

    display_tree(
        model,
        x_train.columns
    )


if __name__ == "__main__":
    main()