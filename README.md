End-to-End Time Series Forecasting System with REST API
Overview
This project is a production-style time series forecasting system that predicts the next 8 weeks of sales for each state using historical sales data.
The system:


Trains multiple forecasting models


Performs feature engineering


Evaluates model performance


Automatically selects the best model


Exposes predictions through a FastAPI REST API



Features


Multi-model forecasting pipeline


State-wise forecasting


Automatic best model selection


Time-series aware train/test split


Feature engineering


REST API deployment using FastAPI


Interactive Swagger documentation


Forecast visualization generation



Forecasting Models Used
The following models were trained and compared:


SARIMA


Facebook Prophet


XGBoost


LSTM (Deep Learning)



Feature Engineering
Implemented features include:


Lag Features


t-1


t-7


t-30




Rolling Statistics


Rolling Mean


Rolling Standard Deviation




Date Features


Day of Week


Month


Week of Year


Quarter




Holiday Flag



Project Structure
forecasting_system/│├── train.py├── app.py├── requirements.txt│├── data/│   └── sales.xlsx│├── models/│   ├── best_model.pkl│   └── metadata.json│├── plots/│└── README.md

Installation
Install dependencies:
pip install -r requirements.txt

Training the Models
Run the training pipeline:
python train.py data/sales.xlsx
This will:


clean the dataset


generate features


train all models


evaluate MAE/RMSE


select best models


save plots


save trained models



Running the API
Start the FastAPI server:
uvicorn app:app --reload
API runs at:
http://127.0.0.1:8000
Swagger API documentation:
http://127.0.0.1:8000/docs

API Endpoints
EndpointDescription/API status/healthHealth check/statesAvailable states/model-info/{state}Best model and metrics/predict/{state}8-week forecast

Example Prediction Request
GET /predict/Texas
Example response:
{  "state": "Texas",  "best_model": "LSTM",  "forecast": [    {      "date": "2026-05-15",      "predicted_sales": 1234567    }  ]}

Evaluation Metrics
Models are evaluated using:


MAE (Mean Absolute Error)


RMSE (Root Mean Squared Error)


The model with the lowest MAE is automatically selected as the best model for each state.

Technologies Used


Python


Pandas


NumPy


Scikit-learn


Statsmodels


Prophet


XGBoost


TensorFlow / Keras


FastAPI


Matplotlib



Output Artifacts
Generated outputs include:


Forecast plots (plots/)


Trained models (models/best_model.pkl)


Model metadata (models/metadata.json)



Author
Krishan V Naikmasur
