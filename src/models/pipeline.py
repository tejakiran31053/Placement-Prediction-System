"""
Shared training pipeline for the Flask app.

Trains Logistic Regression, Decision Tree, Random Forest and Gradient
Boosting once on the placement dataset, caches the fitted models +
preprocessors in memory, and exposes helpers so every page
(/preprocessing, /models, /evaluation, /comparison, /prediction) can
reuse the exact same fitted objects instead of retraining on every
request.
"""

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.data.load_data import load_data
from src.data.preprocess import (
    split_data,
    identify_features,
    handle_missing_values,
    standardize_data,
    one_hot_encode_data,
    ordinal_encode_data,
)

TARGET_COLUMN = "PlacementStatus"
DROP_COLUMNS = ["StudentID", "Salary Package", "IsAnomaly"]

ONE_HOT_FEATURES = [
    "Gender", "City", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs"
]
ORDINAL_FEATURES = ["CollegeTier", "CGPA_Tier"]

# Simple in-memory cache so the (fairly expensive) training step only
# runs once per app run instead of once per request.
_cache = {}


def get_model_definitions():
    """Returns a fresh dict of un-trained model instances."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            criterion="entropy", max_depth=5, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_features="sqrt", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        ),
    }


def build_preprocessed_data():
    """Loads the raw dataset and runs it through the full
    preprocessing pipeline (split -> impute -> scale -> encode)."""
    df = load_data()

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column=TARGET_COLUMN,
        drop_columns=DROP_COLUMNS,
        stratify=True,
    )

    numerical_features, categorical_features = identify_features(X_train)

    X_train, X_test, imputer = handle_missing_values(
        X_train, X_test, numerical_features
    )

    X_train, X_test, scaler = standardize_data(
        X_train, X_test, numerical_features
    )

    X_train, X_test, one_hot_encoder = one_hot_encode_data(
        X_train, X_test, ONE_HOT_FEATURES
    )

    X_train, X_test, ordinal_encoder = ordinal_encode_data(
        X_train, X_test, ORDINAL_FEATURES
    )

    return {
        "raw_df": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "imputer": imputer,
        "scaler": scaler,
        "one_hot_encoder": one_hot_encoder,
        "ordinal_encoder": ordinal_encoder,
    }


def train_all_models(force_refresh=False):
    """Trains every model once and caches the fitted models + metrics.
    Subsequent calls just return the cached dict."""
    if _cache and not force_refresh:
        return _cache

    data = build_preprocessed_data()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    model_defs = get_model_definitions()
    trained_models = {}
    results = {}

    for name, model in model_defs.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        trained_models[name] = model

    best_model_name = max(results, key=lambda n: results[n]["accuracy"])

    _cache.clear()
    _cache.update({
        "data": data,
        "models": trained_models,
        "results": results,
        "best_model_name": best_model_name,
        "feature_columns": X_train.columns.tolist(),
    })

    return _cache


def preprocess_single_record(record: dict):
    """Applies the SAME fitted imputer / scaler / encoders used during
    training to a single raw record (from the prediction form) and
    returns a one-row DataFrame ready for model.predict()."""
    cache = train_all_models()
    data = cache["data"]

    df_row = pd.DataFrame([record])

    numerical_features = data["numerical_features"]
    df_row[numerical_features] = data["imputer"].transform(
        df_row[numerical_features]
    )
    df_row[numerical_features] = data["scaler"].transform(
        df_row[numerical_features]
    )

    one_hot_encoded = data["one_hot_encoder"].transform(
        df_row[ONE_HOT_FEATURES]
    )
    one_hot_df = pd.DataFrame(
        one_hot_encoded,
        columns=data["one_hot_encoder"].get_feature_names_out(ONE_HOT_FEATURES),
        index=df_row.index,
    )

    ordinal_encoded = data["ordinal_encoder"].transform(
        df_row[ORDINAL_FEATURES]
    )
    ordinal_df = pd.DataFrame(
        ordinal_encoded, columns=ORDINAL_FEATURES, index=df_row.index
    )

    df_row = df_row.drop(columns=ONE_HOT_FEATURES + ORDINAL_FEATURES)
    df_row = pd.concat([df_row, one_hot_df, ordinal_df], axis=1)

    # Make sure column order / presence exactly matches what the
    # models were trained on.
    df_row = df_row.reindex(columns=cache["feature_columns"], fill_value=0)

    return df_row


if __name__ == "__main__":
    cache = train_all_models()
    print("\nModel accuracies:")
    for name, metrics in cache["results"].items():
        print(f"  {name}: {metrics['accuracy'] * 100:.2f}%")
    print(f"\nBest model: {cache['best_model_name']}")
