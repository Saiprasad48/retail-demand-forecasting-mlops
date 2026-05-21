from pathlib import Path
import pandas as pd
import numpy as np

PROCESSED_DATA_PATH = Path("data/processed")

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.day
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store_nbr", "family", "date"])
    group_cols = ["store_nbr", "family"]
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df.groupby(group_cols)["sales"].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store_nbr", "family", "date"])
    group_cols = ["store_nbr", "family"]
    for window in [7, 14, 28]:
        df[f"rolling_mean_{window}"] = (
            df.groupby(group_cols)["sales"]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )
    return df

def clean_feature_data(df: pd.DataFrame) -> pd.DataFrame:
    df["onpromotion"] = df["onpromotion"].fillna(0)
    df["transactions"] = df["transactions"].fillna(0)
    df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()
    categorical_cols = [
        "family",
        "city",
        "state",
        "store_type",
        "holiday_type",
        "holiday_description",
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df

def build_features():
    input_path = PROCESSED_DATA_PATH / "train_processed.csv"
    output_path = PROCESSED_DATA_PATH / "train_features.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            "train_processed.csv not found. Run src/data/preprocess.py first."
        )
    df = pd.read_csv(input_path)
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = clean_feature_data(df)
    df.to_csv(output_path, index=False)
    print("Feature engineering completed successfully.")
    print(f"Output path: {output_path}")
    print(f"Shape: {df.shape}")
    print("Columns:")
    print(list(df.columns))
    print("\nSample:")
    print(df.head())

if __name__ == "__main__":
    build_features()