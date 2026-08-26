from sklearn import linear_model
from sklearn.linear_model import (
   LinearRegression,
   Ridge,
   Lasso,
   ElasticNet
)
from sklearn.metrics import(
 mean_absolute_error,
 mean_squared_error,
 r2_score
)

from src.data import load_data

from src.data.preprocess import (
 split_data,
 identify_features,
handle_missing_values,
standardize_data,
one_hot_encode_data,
ordinal_encode_data
)

def create_models():
    models = {
        "LinearRegression":
            LinearRegression(),
        "Ridge Regression":
             Ridge(alpha=1.0),
        "Lasso Regression":
            Lasso(alpha=0.01),
        "ElasticNet Regression":
           ElasticNet(
             alpha=0.01,
             l1_ratio=0.5)
    }
    return models

def train_model(model, X_tarin, y_tarin):
    model.fit(X_tarin, y_tarin)
    return model

def predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

def evaluate(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    return {
        "MAE" :mae,
        "MSE" :mse,
        "RMSE" :rmse,
        "R2" :r2,
    }

def main():

    df = load_data()
    print("/n Orginal Dataset Shape:")
    print(df.shape)