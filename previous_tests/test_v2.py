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

# Modelling and Forecasting
# ==============================================================================
from statsmodels.tsa.statespace.sarimax import SARIMAX
import shap
shap.initjs()

# Warnings configuration
# ==============================================================================
import warnings
warnings.filterwarnings('once')

# Loading the data
# ==============================================================================
data = pd.read_csv('datasets/salesdaily.csv')
data.info()

# Data preparation
# ==============================================================================
data['datum'] = pd.to_datetime(data['datum'])
data = data.set_index('datum')
data = data.asfreq('1D')
data = data.sort_index()
data.head(2)

# Verify that the time series is complete
# ==============================================================================
(data.index == pd.date_range(start=data.index.min(),
                             end=data.index.max(),
                             freq=data.index.freq)).all()

print(f"Number of rows with missing values: {data.isnull().any(axis=1).mean()}")

# if not complete, fill with NaN values
if data.isnull().any(axis=1).mean() > 0.0:
    data.asfreq(freq='1D', fill_value=np.nan)


fig, ax = plt.subplots(figsize=(16,9), facecolor='w')
ax.plot(data.loc[:, ['R06']], label='Testing Set')
""" plt.show() """

# Split the remaining data into train-validation-test (80-20)
# ==============================================================================
number_of_dates = data.shape[0]
index_train = round(70/100 * number_of_dates)
index_validation = index_train + round(15/100 * number_of_dates)

start_train = data.index[0]
end_train = data.index[index_train - 1]

start_validation = data.index[index_train]
end_validation = data.index[index_validation-1]

start_test = data.index[index_validation]

data_train = data.loc[:end_validation, :].copy()
data_test  = data.loc[start_test:, :].copy()

end_test = data_test.index.max()

data_train = data_train.loc[:, ['R06']]
data_test = data_test.loc[:, ['R06']]

# Set the forecast horizon and the lags
# ==============================================================================
steps = 1
lags = 300

# Create forecaster
# ==============================================================================
forecaster = ForecasterAutoreg(
                 regressor = LGBMRegressor(random_state=15926, verbose=-1),
                 lags      = lags
             )

# Train forecaster
# ==============================================================================
forecaster.fit(y=data_train['R06'])

# Backtesting
# ==============================================================================
metric, predictions = backtesting_forecaster(
                          forecaster         = forecaster,
                          y                  = data['R06'],
                          steps              = steps,
                          metric             = 'mean_absolute_error',
                          initial_train_size = len(data_train['R06']),
                          refit              = False,
                          n_jobs             = 'auto',
                          verbose            = True,
                          show_progress      = True
                      )

# Backtesting error
# ==============================================================================
print(f'Backtest error (MAE): {metric}')


# Hyperparameters search
# ==============================================================================
forecaster = ForecasterAutoreg(
                 regressor = LGBMRegressor(random_state=15926, verbose=-1),
                 lags      = lags, # This value will be replaced in the search
             )

# Lags used as predictors
lags_grid = [10, 50, 100, 200, 300, 500, 1000]

# Regressor hyperparameters search space
def search_space(trial):
    search_space  = {
        'n_estimators'  : trial.suggest_int('n_estimators', 800, 1500, step=100),
        'max_depth'     : trial.suggest_int('max_depth', 3, 10, step=1),
        'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.5),
        'reg_alpha'     : trial.suggest_float('reg_alpha', 0, 1, step=0.1),
        'reg_lambda'    : trial.suggest_float('reg_lambda', 0, 1, step=0.1),
    } 
    return search_space

# Perform the search
# ==============================================================================
results_search, frozen_trial = bayesian_search_forecaster(
                                   forecaster         = forecaster,
                                   y                  = data['R06'],
                                   steps              = steps,
                                   metric             = 'mean_absolute_error',
                                   search_space       = search_space,
                                   lags_grid          = lags_grid,
                                   initial_train_size = len(data_train['R06']),
                                   refit              = False,
                                   n_trials           = 300, # Increase for more exhaustive search
                                   random_state       = 123,
                                   return_best        = True,
                                   n_jobs             = 'auto',
                                   verbose            = False,
                                   show_progress      = True
                               )

# Search results
# ==============================================================================
results_search.head(10)

# Backtest final model on test data
# ==============================================================================
metric, predictions = backtesting_forecaster(
                          forecaster         = forecaster,
                          y                  = data['R06'],
                          steps              = steps,
                          metric             = 'mean_absolute_error',
                          initial_train_size = len(data_train['R06']),
                          refit              = False,
                          n_jobs             = 'auto',
                          verbose            = True,
                          show_progress      = True
                      )

print(f"Backtest error (MAE): {metric:.2f}")

# Figure
fig, ax = plt.subplots(figsize=(16,9), facecolor='w')

ax.plot(data_test, label='Testing Set')
ax.plot(predictions, label='Forecast')

# Labels
ax.set_title("Test vs Predictions", fontsize=15, pad=10)
ax.set_ylabel("Number of orders", fontsize=12)
ax.set_xlabel("Date", fontsize=12)

# Legend & Grid
ax.grid(linestyle=":", color='grey')
ax.legend()
plt.show()

st = 1

