CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    prediction_date DATE NOT NULL,
    store_nbr INT NOT NULL,
    family VARCHAR(100) NOT NULL,
    predicted_sales FLOAT NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);