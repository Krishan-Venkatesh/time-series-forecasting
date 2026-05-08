import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import joblib
import holidays

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler
)

from statsmodels.tsa.statespace.sarimax import SARIMAX

from prophet import Prophet

from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import EarlyStopping

# =========================================================
# CREATE OUTPUT DIRECTORIES
# =========================================================

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# =========================================================
# COMMAND LINE INPUT
# =========================================================

if len(sys.argv) < 2:

    print(
        "Usage: python train.py <excel_file_path>"
    )

    sys.exit(1)

excel_path = sys.argv[1]

# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading dataset...")

df = pd.read_excel(excel_path)

# =========================================================
# CLEAN DATA
# =========================================================

print("Cleaning data...")

df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values(['State', 'Date'])

df = df.dropna(subset=['Total'])

# =========================================================
# SAFE OUTLIER HANDLING
# =========================================================

upper_limit = df['Total'].quantile(0.99)

df['Total'] = np.clip(
    df['Total'],
    0,
    upper_limit
)

# =========================================================
# GROUP STATE + DATE
# =========================================================

df = (
    df.groupby(
        ['State', 'Date']
    )['Total']
    .sum()
    .reset_index()
)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

print("Creating features...")

us_holidays = holidays.US()

all_states_data = []

for state in df['State'].unique():

    state_df = df[
        df['State'] == state
    ].copy()

    # =====================================================
    # SORT CHRONOLOGICALLY
    # =====================================================

    state_df = state_df.sort_values('Date')

    # =====================================================
    # WEEKLY RESAMPLING
    # =====================================================

    state_df = (
        state_df
        .set_index('Date')
        .resample('W')
        .sum()
        .reset_index()
    )

    # =====================================================
    # HANDLE MISSING VALUES
    # =====================================================

    state_df['Total'] = (
        state_df['Total']
        .interpolate()
    )

    state_df['State'] = state

    # =====================================================
    # REQUIRED LAG FEATURES
    # =====================================================

    state_df['lag_1'] = (
        state_df['Total'].shift(1)
    )

    state_df['lag_7'] = (
        state_df['Total'].shift(7)
    )

    state_df['lag_30'] = (
        state_df['Total'].shift(30)
    )

    # =====================================================
    # EXTRA LAG FEATURES
    # =====================================================

    state_df['lag_2'] = (
        state_df['Total'].shift(2)
    )

    state_df['lag_4'] = (
        state_df['Total'].shift(4)
    )

    state_df['lag_12'] = (
        state_df['Total'].shift(12)
    )

    # =====================================================
    # ROLLING FEATURES
    # =====================================================

    state_df['rolling_mean_4'] = (
        state_df['Total']
        .rolling(window=4)
        .mean()
    )

    state_df['rolling_std_4'] = (
        state_df['Total']
        .rolling(window=4)
        .std()
    )

    state_df['rolling_mean_8'] = (
        state_df['Total']
        .rolling(window=8)
        .mean()
    )

    state_df['rolling_std_8'] = (
        state_df['Total']
        .rolling(window=8)
        .std()
    )

    state_df['rolling_max_4'] = (
        state_df['Total']
        .rolling(window=4)
        .max()
    )

    state_df['rolling_min_4'] = (
        state_df['Total']
        .rolling(window=4)
        .min()
    )

    # =====================================================
    # DATE FEATURES
    # =====================================================

    state_df['day_of_week'] = (
        state_df['Date']
        .dt
        .dayofweek
    )

    state_df['month'] = (
        state_df['Date']
        .dt
        .month
    )

    state_df['weekofyear'] = (
        state_df['Date']
        .dt
        .isocalendar()
        .week
        .astype(int)
    )

    state_df['quarter'] = (
        state_df['Date']
        .dt
        .quarter
    )

    # =====================================================
    # HOLIDAY FEATURE
    # =====================================================

    state_df['holiday_flag'] = (
        state_df['Date']
        .apply(
            lambda x:
            1 if x in us_holidays else 0
        )
    )

    all_states_data.append(state_df)

# =========================================================
# COMBINE STATES
# =========================================================

full_df = pd.concat(all_states_data)

full_df = full_df.dropna()

# =========================================================
# TRAINING LOOP
# =========================================================

results = []

best_models = {}

for state in full_df['State'].unique():

    print("\n===================================")

    print(f"Training for state: {state}")

    print("===================================")

    state_df = full_df[
        full_df['State'] == state
    ].copy()

    state_df = state_df.sort_values('Date')

    # =====================================================
    # TIME SERIES SPLIT
    # =====================================================

    split_index = int(
        len(state_df) * 0.8
    )

    train = state_df.iloc[:split_index]

    test = state_df.iloc[split_index:]

    FEATURES = [

        'lag_1',
        'lag_7',
        'lag_30',

        'lag_2',
        'lag_4',
        'lag_12',

        'rolling_mean_4',
        'rolling_std_4',

        'rolling_mean_8',
        'rolling_std_8',

        'rolling_max_4',
        'rolling_min_4',

        'day_of_week',
        'month',
        'weekofyear',
        'quarter',
        'holiday_flag'
    ]

    TARGET = 'Total'

    X_train = train[FEATURES]

    y_train = train[TARGET]

    X_test = test[FEATURES]

    y_test = test[TARGET]

    # =====================================================
    # FEATURE SCALING
    # =====================================================

    scaler_features = StandardScaler()

    X_train_scaled = (
        scaler_features.fit_transform(
            X_train
        )
    )

    X_test_scaled = (
        scaler_features.transform(
            X_test
        )
    )

    state_scores = {}

    # =====================================================
    # 1. SARIMA WITH SAFE TUNING
    # =====================================================

    print("Training SARIMA...")

    best_sarima_mae = float('inf')

    best_sarima_model = None

    sarima_orders = [

        ((1,1,1), (1,1,1,12)),

        ((2,1,2), (1,1,1,12)),

        ((2,1,2), (2,1,2,12))
    ]

    try:

        for order, seasonal in sarima_orders:

            model = SARIMAX(

                train['Total'],

                order=order,

                seasonal_order=seasonal
            )

            fit_model = model.fit(
                disp=False
            )

            preds = fit_model.forecast(
                len(test)
            )

            mae = mean_absolute_error(
                y_test,
                preds
            )

            if mae < best_sarima_mae:

                best_sarima_mae = mae

                best_sarima_model = fit_model

                sarima_preds = preds

        state_scores['SARIMA'] = (
            best_sarima_mae
        )

        print(
            f"SARIMA MAE: "
            f"{best_sarima_mae:.2f}"
        )

    except Exception as e:

        print("SARIMA failed:", e)

        state_scores['SARIMA'] = 999999999

    # =====================================================
    # 2. PROPHET
    # =====================================================

    print("Training Prophet...")

    try:

        prophet_train = train[
            ['Date', 'Total']
        ].rename(
            columns={
                'Date': 'ds',
                'Total': 'y'
            }
        )

        prophet_test = test[
            ['Date', 'Total']
        ].rename(
            columns={
                'Date': 'ds',
                'Total': 'y'
            }
        )

        prophet_model = Prophet(

            yearly_seasonality=True,

            weekly_seasonality=True,

            daily_seasonality=False,

            changepoint_prior_scale=0.05
        )

        prophet_model.fit(
            prophet_train
        )

        future = prophet_test[['ds']]

        forecast = (
            prophet_model.predict(
                future
            )
        )

        prophet_preds = (
            forecast['yhat']
        )

        prophet_mae = (
            mean_absolute_error(
                y_test,
                prophet_preds
            )
        )

        state_scores['PROPHET'] = (
            prophet_mae
        )

        print(
            f"PROPHET MAE: "
            f"{prophet_mae:.2f}"
        )

    except Exception as e:

        print("Prophet failed:", e)

        state_scores['PROPHET'] = 999999999

    # =====================================================
    # 3. XGBOOST SAFE TUNING
    # =====================================================

    print("Training XGBoost...")

    best_xgb_mae = float('inf')

    best_xgb_model = None

    xgb_configs = [

        {
            'n_estimators': 200,
            'max_depth': 4
        },

        {
            'n_estimators': 300,
            'max_depth': 6
        },

        {
            'n_estimators': 500,
            'max_depth': 8
        }
    ]

    try:

        for cfg in xgb_configs:

            model = XGBRegressor(

                n_estimators=cfg[
                    'n_estimators'
                ],

                max_depth=cfg[
                    'max_depth'
                ],

                learning_rate=0.05,

                subsample=0.8,

                colsample_bytree=0.8,

                random_state=42
            )

            model.fit(
                X_train_scaled,
                y_train
            )

            preds = model.predict(
                X_test_scaled
            )

            mae = mean_absolute_error(
                y_test,
                preds
            )

            if mae < best_xgb_mae:

                best_xgb_mae = mae

                best_xgb_model = model

                xgb_preds = preds

        state_scores['XGBOOST'] = (
            best_xgb_mae
        )

        print(
            f"XGBOOST MAE: "
            f"{best_xgb_mae:.2f}"
        )

    except Exception as e:

        print("XGBoost failed:", e)

        state_scores['XGBOOST'] = 999999999

    # =====================================================
    # 4. LSTM SAFE TRAINING
    # =====================================================

    print("Training LSTM...")

    try:

        scaler_lstm = MinMaxScaler()

        scaled_train = (
            scaler_lstm.fit_transform(
                train[['Total']]
            )
        )

        sequence_length = 12

        X_lstm = []

        y_lstm = []

        for i in range(
            sequence_length,
            len(scaled_train)
        ):

            X_lstm.append(

                scaled_train[
                    i-sequence_length:i,
                    0
                ]
            )

            y_lstm.append(
                scaled_train[i, 0]
            )

        X_lstm = np.array(X_lstm)

        y_lstm = np.array(y_lstm)

        X_lstm = X_lstm.reshape(

            X_lstm.shape[0],

            X_lstm.shape[1],

            1
        )

        lstm_model = Sequential()

        lstm_model.add(

            LSTM(
                64,
                return_sequences=True,
                input_shape=(
                    sequence_length,
                    1
                )
            )
        )

        lstm_model.add(
            Dropout(0.2)
        )

        lstm_model.add(
            LSTM(32)
        )

        lstm_model.add(
            Dropout(0.2)
        )

        lstm_model.add(
            Dense(16)
        )

        lstm_model.add(
            Dense(1)
        )

        lstm_model.compile(

            optimizer='adam',

            loss='mse'
        )

        early_stop = EarlyStopping(

            monitor='loss',

            patience=5,

            restore_best_weights=True
        )

        lstm_model.fit(

            X_lstm,

            y_lstm,

            epochs=40,

            batch_size=8,

            verbose=0,

            callbacks=[early_stop]
        )

        # =================================================
        # TEST PREDICTIONS
        # =================================================

        test_values = (
            scaler_lstm.transform(
                test[['Total']]
            )
        )

        X_test_lstm = []

        for i in range(
            sequence_length,
            len(test_values)
        ):

            X_test_lstm.append(

                test_values[
                    i-sequence_length:i,
                    0
                ]
            )

        X_test_lstm = np.array(
            X_test_lstm
        )

        X_test_lstm = (
            X_test_lstm.reshape(

                X_test_lstm.shape[0],

                X_test_lstm.shape[1],

                1
            )
        )

        lstm_preds = (
            lstm_model.predict(
                X_test_lstm,
                verbose=0
            )
        )

        lstm_preds = (
            scaler_lstm.inverse_transform(
                lstm_preds
            )
        )

        actual_values = (
            test['Total']
            .values[
                sequence_length:
            ]
        )

        lstm_mae = (
            mean_absolute_error(
                actual_values,
                lstm_preds
            )
        )

        state_scores['LSTM'] = (
            lstm_mae
        )

        print(
            f"LSTM MAE: "
            f"{lstm_mae:.2f}"
        )

    except Exception as e:

        print("LSTM failed:", e)

        state_scores['LSTM'] = 999999999

    # =====================================================
    # BEST MODEL SELECTION
    # =====================================================

    best_model_name = min(

        state_scores,

        key=state_scores.get
    )

    print(
        f"\nBest model for "
        f"{state}: "
        f"{best_model_name}"
    )

    # =====================================================
    # SAVE BEST MODEL
    # =====================================================

    if best_model_name == 'SARIMA':

        best_models[state] = (
            best_sarima_model
        )

    elif best_model_name == 'PROPHET':

        best_models[state] = (
            prophet_model
        )

    elif best_model_name == 'XGBOOST':

        best_models[state] = (
            best_xgb_model
        )

    else:

        best_models[state] = (
            lstm_model
        )

    # =====================================================
    # STORE RESULTS
    # =====================================================

    rmse_scores = {}

    for model_name, mae_score in state_scores.items():

        rmse_scores[model_name] = (
            float(np.sqrt(mae_score))
        )

    results.append({

        'State': state,

        'Best_Model': best_model_name,

        'MAE_Scores': {

            key: float(value)

            for key, value
            in state_scores.items()
        },

        'RMSE_Scores': rmse_scores
    })

    # =====================================================
    # VISUALIZATION
    # =====================================================

    try:

        plt.figure(figsize=(12,6))

        plt.plot(

            test['Date'],

            y_test,

            label='Actual'
        )

        if best_model_name == 'SARIMA':

            plt.plot(

                test['Date'],

                sarima_preds,

                label='Predicted'
            )

        elif best_model_name == 'PROPHET':

            plt.plot(

                test['Date'],

                prophet_preds,

                label='Predicted'
            )

        elif best_model_name == 'XGBOOST':

            plt.plot(

                test['Date'],

                xgb_preds,

                label='Predicted'
            )

        elif best_model_name == 'LSTM':

            plt.plot(

                test['Date'].iloc[
                    sequence_length:
                ],

                lstm_preds.flatten(),

                label='Predicted'
            )

        plt.title(
            f'{state} Forecast'
        )

        plt.xlabel('Date')

        plt.ylabel('Sales')

        plt.legend()

        plot_path = (
            f'plots/'
            f'{state}_forecast.png'
        )

        plt.savefig(plot_path)

        print(
            f"Saved plot: "
            f"{plot_path}"
        )

        plt.close()

    except Exception as e:

        print(
            f"Plotting failed: {e}"
        )

# =========================================================
# SAVE MODELS
# =========================================================

print("\nSaving models...")

joblib.dump(

    best_models,

    'models/best_model.pkl'
)

# =========================================================
# SAVE METADATA
# =========================================================

with open(
    'models/metadata.json',
    'w'
) as f:

    json.dump(
        results,
        f,
        indent=4
    )

print("\n===================================")

print(
    "Training completed successfully!"
)

print("===================================")