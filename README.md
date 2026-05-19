# Retail Demand Forecasting with MLOps

This project builds an end-to-end retail demand forecasting system using historical sales data.

## Goal

Predict future product demand or sales using historical retail data and deploy the model using an MLOps-style workflow.

## Tech Stack

- Python
- Pandas
- Scikit-learn
- XGBoost / LightGBM
- PostgreSQL
- Docker
- FastAPI
- MLflow

## Project Structure

```text
retail-demand-forecasting-mlops/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── api/
│   └── database/
├── docker-compose.yml
├── requirements.txt
├── config.yaml
└── README.md