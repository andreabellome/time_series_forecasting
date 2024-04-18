# Data manipulation
# ==============================================================================
import numpy as np
import pandas as pd

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

# Set the forecast horizon and the lags
# ==============================================================================
steps = 1
lags = 1850

# Hyperparameters
# ==============================================================================
lgbm_hyperparameters = {
    'n_estimators': 1200,       # Number of boosting rounds
    'learning_rate': 0.0100786818966835,  # Learning rate
    'max_depth': 5,             # Maximum tree depth
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
}

# Create forecaster
# ==============================================================================
forecaster = ForecasterAutoreg(
                 regressor = LGBMRegressor(random_state=15926, verbose=-1, **lgbm_hyperparameters),
                 lags      = lags
             )

# Train forecaster
# ==============================================================================
forecaster.fit(y=data.loc[:end_train, 'Load'])
forecaster

steps = 1

# Backtest final model on test data for two steps ahead predictions
# ==============================================================================
metric, two_steps_predictions = backtesting_forecaster(
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

# Obtain the two steps ahead prediction from backtesting
two_steps_prediction = two_steps_predictions[0]

# Make two steps ahead prediction directly using forecaster.predict
direct_two_steps_prediction = forecaster.predict(steps=steps)[0]

# Print or use both predictions
print("Two steps ahead prediction from backtesting:", two_steps_prediction)
print("Direct two steps ahead prediction using forecaster.predict:", direct_two_steps_prediction)

st = 1