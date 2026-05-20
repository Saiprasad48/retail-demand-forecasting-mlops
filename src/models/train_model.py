from pathlib import Path
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error


PROCESSED_DATA_PATH = Path("data/processed")
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)


def load_data():
    file_path = PROCESSED_DATA_PATH / "train_features.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            "train_features.csv not found. Run src/features/build_features.py first."
        )

    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])

    return df


def encode_categorical_columns(df):
    categorical_cols = [
        "family",
        "city",
        "state",
        "store_type",
        "holiday_type",
        "holiday_description",
    ]

    encoders = {}

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            codes, uniques = pd.factorize(df[col])
            df[col] = codes
            encoders[col] = list(uniques)

    return df, encoders


def train_test_split_by_date(df, test_days=30):
    max_date = df["date"].max()
    cutoff_date = max_date - pd.Timedelta(days=test_days)

    train_df = df[df["date"] <= cutoff_date].copy()
    test_df = df[df["date"] > cutoff_date].copy()

    return train_df, test_df, cutoff_date


def get_features():
    return [
        "store_nbr",
        "family",
        "onpromotion",
        "city",
        "state",
        "store_type",
        "cluster",
        "dcoilwtico",
        "transactions",
        "is_holiday",
        "holiday_type",
        "holiday_description",
        "day",
        "week",
        "month",
        "year",
        "day_of_week",
        "is_weekend",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
    ]


def calculate_metrics(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))

    return {
        "mae": mae,
        "rmse": rmse,
        "rmsle": rmsle,
    }


def train_model():
    df = load_data()

    df, encoders = encode_categorical_columns(df)

    train_df, test_df, cutoff_date = train_test_split_by_date(df, test_days=30)

    features = get_features()
    target = "sales"

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    mlflow.set_experiment("retail-demand-forecasting")

    with mlflow.start_run(run_name="lightgbm_v1"):
        params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": -1,
            "num_leaves": 64,
            "random_state": 42,
            "n_jobs": -1,
        }

        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        metrics = calculate_metrics(y_test, predictions)

        mlflow.log_params(params)
        mlflow.log_metric("mae", metrics["mae"])
        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("rmsle", metrics["rmsle"])
        mlflow.log_param("cutoff_date", str(cutoff_date.date()))
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))

        mlflow.sklearn.log_model(model, "model")

        model_file = MODEL_PATH / "lightgbm_model.joblib"
        encoder_file = MODEL_PATH / "encoders.json"
        feature_file = MODEL_PATH / "features.json"

        joblib.dump(model, model_file)

        with open(encoder_file, "w") as f:
            json.dump(encoders, f, indent=4)

        with open(feature_file, "w") as f:
            json.dump(features, f, indent=4)

        print("Model training completed successfully.")
        print(f"Cutoff date: {cutoff_date.date()}")
        print(f"Train rows: {len(train_df)}")
        print(f"Test rows: {len(test_df)}")
        print("\nMetrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

        print(f"\nModel saved to: {model_file}")
        print(f"Encoders saved to: {encoder_file}")
        print(f"Features saved to: {feature_file}")


if __name__ == "__main__":
    train_model()