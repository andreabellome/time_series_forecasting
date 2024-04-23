# Data manipulation
# ==============================================================================
import numpy as np
import pandas as pd
import pickle
from astral.sun import sun
from astral import LocationInfo
from skforecast.datasets import fetch_dataset

# Plots
# ==============================================================================
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as poff

pio.renderers.default = 'notebook' 
pio.templates.default = "seaborn"
poff.init_notebook_mode(connected=True)
plt.style.use('seaborn-v0_8-darkgrid')

# Modelling and Forecasting
# ==============================================================================
from lightgbm import LGBMRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import bayesian_search_forecaster
from skforecast.model_selection import backtesting_forecaster
import shap
shap.initjs()

from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.metrics import mean_absolute_error

# Warnings configuration
# ==============================================================================
import warnings
warnings.filterwarnings('once')


# Loading the data
# ==============================================================================
data = pd.read_csv('datasets/load.csv')
data.info()

# Data preparation
# ==============================================================================
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
data = data.set_index('Date')
data = data.asfreq('1D')
data = data.sort_index()
data.head(5) 

# Verify that the time series is complete
# ==============================================================================
(data.index == pd.date_range(start=data.index.min(),
                             end=data.index.max(),
                             freq=data.index.freq)).all()

print(f"Number of rows with missing values: {data.isnull().any(axis=1).mean()}")

# if not complete, fill with NaN values
if data.isnull().any(axis=1).mean() > 0.0:
    data.asfreq(freq='1D', fill_value=np.nan)

    
# Split the remaining data into train-validation-test
# ==============================================================================
data = data.loc['2006-01-01': '2019-12-31'].copy()
start_train = '2006-01-01'
end_train = '2018-12-31'
start_test = '2019-01-01'
data_train = data.loc[start_train:end_train, :].copy()
data_test  = data.loc[start_test:, :].copy()

print(f"Train dates      : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
print(f"Test dates       : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")

calendar_features = pd.DataFrame(index=data.index)
calendar_features['month'] = calendar_features.index.month
calendar_features['week_of_year'] = calendar_features.index.isocalendar().week
calendar_features['week_day'] = calendar_features.index.day_of_week + 1
calendar_features['hour_day'] = calendar_features.index.hour + 1

# Sunlight features
# ==============================================================================
location = LocationInfo(
    "Rome",
    "Italy",
    latitude=41.902782,
    longitude=12.496366,
    timezone='Europe/Rome'
)
sunrise_hour = [
    sun(location.observer, date=date, tzinfo=location.timezone)['sunrise'].hour
    for date in data.index
]
sunset_hour = [
    sun(location.observer, date=date, tzinfo=location.timezone)['sunset'].hour
    for date in data.index
]
sun_light_features = pd.DataFrame({
                         'sunrise_hour': sunrise_hour,
                         'sunset_hour': sunset_hour}, 
                         index = data.index
                     )

sun_light_features['daylight_hours'] = (
    sun_light_features['sunset_hour'] - sun_light_features['sunrise_hour']
)
sun_light_features['is_daylight'] = np.where(
                                        (data.index.hour >= sun_light_features['sunrise_hour']) & \
                                        (data.index.hour < sun_light_features['sunset_hour']),
                                        1,
                                        0
                                    )

st = 1

# Set the forecast horizon and the lags
# ==============================================================================
steps = 365
lags = 1850

# Hyperparameters
# ==============================================================================
""" lgbm_hyperparameters = {
    'n_estimators': 1200,       # Number of boosting rounds
    'learning_rate': 0.0100786818966835,  # Learning rate
    'max_depth': 5,             # Maximum tree depth
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
} """

lgbm_hyperparameters = {
    'n_estimators': 800,       # Number of boosting rounds
    'learning_rate': 0.22112275403152545,  # Learning rate
    'max_depth': 6,             # Maximum tree depth
    'reg_alpha': 0.5,
    'reg_lambda': 0.4,
}

# Create forecaster
# ==============================================================================
forecaster = ForecasterAutoreg(
                 regressor = LGBMRegressor(random_state=15926, verbose=-1, **lgbm_hyperparameters),
                 lags      = lags
             )

# Train forecaster
# ==============================================================================

data_train = data.loc[:end_train, 'Load']

forecaster.fit(y=data_train)
forecaster

predictionManual = forecaster.predict(steps, data_train.iloc[-lags:])
error = ( predictionManual - data_test.loc[:, 'Load'].iloc[:steps] ).abs().mean()

st = 1

with open('forecaster_params.pkl', 'wb') as file:
    pickle.dump(forecaster, file)


st = 1

# Backtest final model on test data for two steps ahead predictions
# ==============================================================================
metric, predictions = backtesting_forecaster(
    forecaster=forecaster,
    y=data.loc[:, 'Load'],
    steps=steps,  # Set steps to 2 for two steps ahead predictions
    metric='mean_absolute_error',
    initial_train_size=len(data[:end_train]),
    refit=False,
    n_jobs='auto',
    verbose=False,
    show_progress=True
)

st = 1

# Plot predictions vs real value
# ======================================================================================
fig = go.Figure()
trace1 = go.Scatter(x=data_test.index, y=data_test['Load'], name="to predict", mode="lines")
trace2 = go.Scatter(x=predictions.index, y=predictions['pred'], name="prediction", mode="lines")

fig.add_trace(trace1)
fig.add_trace(trace2)

fig.update_layout(
    title="Real value vs predicted in test data (Wh)",
    xaxis_title="Date time",
    yaxis_title="Load",
    width=800,
    height=400,
    margin=dict(l=20, r=20, t=35, b=20),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.1,
        xanchor="left",
        x=0.001
    )
)
fig.show()
fig.write_image("predictions_vs_real.png")

st = 1