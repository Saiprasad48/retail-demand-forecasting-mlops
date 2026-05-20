import os
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


load_dotenv()


DB_USER = os.getenv("DB_USER", "retail_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "retail_pass")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "retail_forecasting")


DATABASE_URL = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL)


def test_connection():
    with engine.connect() as connection:
        result = connection.execute(text("SELECT version();"))
        print("Database connected successfully.")
        print(result.fetchone()[0])


def save_prediction(prediction_date, store_nbr, family, predicted_sales, model_name):
    query = text(
        """
        INSERT INTO predictions (
            prediction_date,
            store_nbr,
            family,
            predicted_sales,
            model_name,
            created_at
        )
        VALUES (
            :prediction_date,
            :store_nbr,
            :family,
            :predicted_sales,
            :model_name,
            :created_at
        )
        RETURNING id;
        """
    )

    with engine.begin() as connection:
        result = connection.execute(
            query,
            {
                "prediction_date": prediction_date,
                "store_nbr": store_nbr,
                "family": family,
                "predicted_sales": predicted_sales,
                "model_name": model_name,
                "created_at": datetime.now(),
            },
        )

        prediction_id = result.fetchone()[0]

    return prediction_id


def get_recent_predictions(limit=10):
    query = text(
        """
        SELECT
            id,
            prediction_date,
            store_nbr,
            family,
            predicted_sales,
            model_name,
            created_at
        FROM predictions
        ORDER BY created_at DESC
        LIMIT :limit;
        """
    )

    with engine.connect() as connection:
        result = connection.execute(query, {"limit": limit})
        rows = result.fetchall()

    return rows


if __name__ == "__main__":
    test_connection()