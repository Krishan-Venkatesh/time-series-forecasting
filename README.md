# End-to-End Time Series Forecasting System with REST API

## Overview

This project is a production-style time series forecasting system that predicts the next 8 weeks of sales for each state using historical sales data.

The system:
- Trains multiple forecasting models
- Performs feature engineering
- Evaluates model performance
- Automatically selects the best model
- Exposes predictions through a FastAPI REST API

---

## Features

- Multi-model forecasting pipeline
- State-wise forecasting
- Automatic best model selection
- Time-series aware train/test split
- Feature engineering
- REST API deployment using FastAPI
- Interactive Swagger documentation
- Forecast visualization generation

---

## Forecasting Models Used

The following models were trained and compared:

1. SARIMA
2. Facebook Prophet
3. XGBoost
4. LSTM (Deep Learning)

---

## Feature Engineering

Implemented features include:

### Lag Features
- t-1
- t-7
- t-30

### Rolling Statistics
- Rolling Mean
- Rolling Standard Deviation

### Date Features
- Day of Week
- Month
- Week of Year
- Quarter

### Holiday Flag

---

## Project Structure

```text
forecasting_system/
│
├── train.py
├── app.py
├── requirements.txt
│
├── data/
│   └── sales.xlsx
│
├── models/
│   ├── best_model.pkl
│   └── metadata.json
│
├── plots/
│
└── README.md
