from pathlib import Path
import pandas as pd


RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")


def load_raw_data():
    train = pd.read_csv(RAW_DATA_PATH / "train.csv")
    stores = pd.read_csv(RAW_DATA_PATH / "stores.csv")
    oil = pd.read_csv(RAW_DATA_PATH / "oil.csv")
    transactions = pd.read_csv(RAW_DATA_PATH / "transactions.csv")
    holidays = pd.read_csv(RAW_DATA_PATH / "holidays_events.csv")

    return train, stores, oil, transactions, holidays


def preprocess_holidays(holidays):
    holidays["date"] = pd.to_datetime(holidays["date"])

    holidays = holidays[holidays["transferred"] == False]

    national_holidays = holidays[holidays["locale"] == "National"].copy()

    national_holidays = national_holidays[["date", "type", "description"]]
    national_holidays = national_holidays.rename(
        columns={
            "type": "holiday_type",
            "description": "holiday_description",
        }
    )

    national_holidays["is_holiday"] = 1

    national_holidays = national_holidays.drop_duplicates(subset=["date"])

    return national_holidays


def preprocess_data():
    train, stores, oil, transactions, holidays = load_raw_data()

    train["date"] = pd.to_datetime(train["date"])
    oil["date"] = pd.to_datetime(oil["date"])
    transactions["date"] = pd.to_datetime(transactions["date"])

    stores = stores.rename(columns={"type": "store_type"})

    holidays_processed = preprocess_holidays(holidays)

    df = train.merge(stores, on="store_nbr", how="left")

    df = df.merge(oil, on="date", how="left")

    df = df.merge(
        transactions,
        on=["date", "store_nbr"],
        how="left"
    )

    df = df.merge(
        holidays_processed,
        on="date",
        how="left"
    )

    df["transactions"] = df["transactions"].fillna(0)
    df["is_holiday"] = df["is_holiday"].fillna(0)
    df["holiday_type"] = df["holiday_type"].fillna("None")
    df["holiday_description"] = df["holiday_description"].fillna("None")

    df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()

    df = df.sort_values(["store_nbr", "family", "date"])

    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DATA_PATH / "train_processed.csv"
    df.to_csv(output_path, index=False)

    print("Processed data saved successfully.")
    print(f"Output path: {output_path}")
    print(f"Shape: {df.shape}")
    print("Columns:")
    print(list(df.columns))
    print("\nSample:")
    print(df.head())


if __name__ == "__main__":
    preprocess_data()