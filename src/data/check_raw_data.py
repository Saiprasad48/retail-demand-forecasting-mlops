from pathlib import Path
import pandas as pd


RAW_DATA_PATH = Path("data/raw")

REQUIRED_FILES = [
    "train.csv",
    "test.csv",
    "stores.csv",
    "transactions.csv",
    "holidays_events.csv",
    "oil.csv",
]


def check_files_exist():
    missing_files = []

    for file_name in REQUIRED_FILES:
        file_path = RAW_DATA_PATH / file_name

        if not file_path.exists():
            missing_files.append(file_name)

    if missing_files:
        print("Missing files:")
        for file_name in missing_files:
            print(f"- {file_name}")
        return False

    print("All required raw data files are available.")
    return True


def preview_files():
    for file_name in REQUIRED_FILES:
        file_path = RAW_DATA_PATH / file_name
        df = pd.read_csv(file_path)

        print("\n" + "=" * 60)
        print(f"File: {file_name}")
        print(f"Shape: {df.shape}")
        print("Columns:")
        print(list(df.columns))
        print("Sample:")
        print(df.head())


if __name__ == "__main__":
    files_available = check_files_exist()

    if files_available:
        preview_files()