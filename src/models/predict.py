from pathlib import Path
import json
import joblib
import pandas as pd


MODEL_PATH = Path("models")
PROCESSED_DATA_PATH = Path("data/processed")


def load_artifacts():
    model = joblib.load(MODEL_PATH / "lightgbm_model.joblib")

    with open(MODEL_PATH / "features.json", "r") as f:
        features = json.load(f)

    with open(MODEL_PATH / "encoders.json", "r") as f:
        encoders = json.load(f)

    return model, features, encoders


def encode_value(value, encoder_values):
    value = str(value)

    if value in encoder_values:
        return encoder_values.index(value)

    return -1


def get_store_metadata(df, store_nbr):
    store_rows = df[df["store_nbr"] == store_nbr]

    if store_rows.empty:
        raise ValueError(f"Store number {store_nbr} not found in historical data.")

    latest_row = store_rows.iloc[-1]

    return {
        "city": latest_row["city"],
        "state": latest_row["state"],
        "store_type": latest_row["store_type"],
        "cluster": latest_row["cluster"],
    }


def get_holiday_features(df, target_date):
    target_date = pd.to_datetime(target_date)

    holiday_rows = df[df["date"] == target_date]

    if holiday_rows.empty:
        return {
            "is_holiday": 0,
            "holiday_type": "None",
            "holiday_description": "None",
        }

    row = holiday_rows.iloc[0]

    return {
        "is_holiday": row.get("is_holiday", 0),
        "holiday_type": row.get("holiday_type", "None"),
        "holiday_description": row.get("holiday_description", "None"),
    }


def get_time_features(target_date):
    target_date = pd.to_datetime(target_date)

    return {
        "day": target_date.day,
        "week": int(target_date.isocalendar().week),
        "month": target_date.month,
        "year": target_date.year,
        "day_of_week": target_date.dayofweek,
        "is_weekend": int(target_date.dayofweek in [5, 6]),
    }


def get_sales_history_features(df, store_nbr, family, target_date):
    target_date = pd.to_datetime(target_date)

    history = df[
        (df["store_nbr"] == store_nbr)
        & (df["family"] == family)
        & (df["date"] < target_date)
    ].sort_values("date")

    if history.empty:
        raise ValueError(
            f"No sales history found for store {store_nbr} and family {family}."
        )

    features = {}

    for lag in [1, 7, 14, 28]:
        if len(history) >= lag:
            features[f"lag_{lag}"] = history.iloc[-lag]["sales"]
        else:
            features[f"lag_{lag}"] = 0

    for window in [7, 14, 28]:
        features[f"rolling_mean_{window}"] = history["sales"].tail(window).mean()

    return features


def get_external_features(df, store_nbr, target_date):
    target_date = pd.to_datetime(target_date)

    past_rows = df[df["date"] <= target_date].sort_values("date")

    if past_rows.empty:
        dcoilwtico = 0
    else:
        dcoilwtico = past_rows.iloc[-1].get("dcoilwtico", 0)

    store_past_rows = df[
        (df["store_nbr"] == store_nbr)
        & (df["date"] < target_date)
    ].sort_values("date")

    if store_past_rows.empty:
        transactions = 0
    else:
        transactions = store_past_rows.iloc[-1].get("transactions", 0)

    return {
        "dcoilwtico": dcoilwtico,
        "transactions": transactions,
    }


def create_prediction_row(store_nbr, family, target_date, onpromotion):
    processed_file = PROCESSED_DATA_PATH / "train_processed.csv"

    if not processed_file.exists():
        raise FileNotFoundError(
            "train_processed.csv not found. Run src/data/preprocess.py first."
        )

    df = pd.read_csv(processed_file)
    df["date"] = pd.to_datetime(df["date"])

    row = {
        "store_nbr": store_nbr,
        "family": family,
        "onpromotion": onpromotion,
    }

    row.update(get_store_metadata(df, store_nbr))
    row.update(get_holiday_features(df, target_date))
    row.update(get_time_features(target_date))
    row.update(get_sales_history_features(df, store_nbr, family, target_date))
    row.update(get_external_features(df, store_nbr, target_date))

    return row


def predict_sales(store_nbr, family, target_date, onpromotion=0):
    model, features, encoders = load_artifacts()

    row = create_prediction_row(
        store_nbr=store_nbr,
        family=family,
        target_date=target_date,
        onpromotion=onpromotion,
    )

    for col, encoder_values in encoders.items():
        if col in row:
            row[col] = encode_value(row[col], encoder_values)

    input_df = pd.DataFrame([row])
    input_df = input_df[features]

    prediction = model.predict(input_df)[0]
    prediction = max(float(prediction), 0)

    return prediction


if __name__ == "__main__":
    result = predict_sales(
        store_nbr=1,
        family="GROCERY I",
        target_date="2017-08-16",
        onpromotion=10,
    )

    print(f"Predicted sales: {result:.2f}")