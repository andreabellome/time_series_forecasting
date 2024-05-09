# Data manipulation
# ==============================================================================
import numpy as np
import pandas as pd
from my_app.functions.processAndPlotData import processAndPlotData

# Plots
# ==============================================================================
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as poff
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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
df_final = pd.read_csv('datasets/df_final_electricity_price.csv')
df_final.info()

instancePlotData = processAndPlotData()

# Convert time to datetime object and set it as index
df_final['time'] = pd.to_datetime(df_final['time'],
                                  utc=True, 
                            infer_datetime_format = True)
df_final = df_final.set_index('time')
df_final = df_final.asfreq('60min')
df_final = df_final.sort_index()
df_final.head(2)

""" fig1 = instancePlotData.plot_raw_data( df_final, 'total load actual', 'Load', 'Time', 'Load [MWh]', usingPlt=True )
fig1.show()

fig2 = instancePlotData.plot_distribution_by_week(df_final, 'total load actual')
fig2.write_html('fig2.html') """

data = df_final.copy()

# Split the remaining data into train-validation-test
# ==============================================================================
end_train = '2017-12-31 23:00:00'
start_test = '2018-01-01 00:00:00'
end_test = '2018-12-31 23:00:00'
data_train = data.loc[:end_train, :].copy()
data_test  = data.loc[start_test:end_test, :].copy()

print(f"Train dates      : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
print(f"Test dates       : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")

columnName = 'total load actual'

# Set the forecast horizon and the lags
# ==============================================================================
steps = 24
lags = 24

# Create forecaster
# ==============================================================================
forecaster = ForecasterAutoreg(
                 regressor = LGBMRegressor(random_state=15926, verbose=-1),
                 lags      = lags
             )

# Train forecaster
# ==============================================================================
forecaster.fit(y=data.loc[:end_train, columnName])

# Backtesting
# ==============================================================================
metric, predictions = backtesting_forecaster(
                          forecaster         = forecaster,
                          y                  = data[columnName],
                          steps              = steps,
                          metric             = 'mean_absolute_error',
                          initial_train_size = len(data.loc[:end_train]),
                          refit              = False,
                          n_jobs             = 'auto',
                          verbose            = True,
                          show_progress      = True
                      )

# Backtesting error
# ==============================================================================
print(f'Backtest error (MAE): {metric}')

""" # Plot predictions vs real value
# ======================================================================================
fig = go.Figure()
trace1 = go.Scatter(x=data_test.index, y=data_test[columnName], name="test", mode="lines")
trace2 = go.Scatter(x=predictions.index, y=predictions['pred'], name="prediction", mode="lines")
fig.add_trace(trace1)
fig.add_trace(trace2)
fig.update_layout(
    title="Real value vs predicted in test data",
    xaxis_title="Date time",
    yaxis_title="Demand",
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
fig.write_html('fig1.html') """

""" # Hyperparameters search
# ==============================================================================
forecaster = ForecasterAutoreg(
                 regressor = LGBMRegressor(random_state=15926, verbose=-1),
                 lags      = lags, # This value will be replaced in the search
             )

# Lags used as predictors
lags_grid = [lags, [1, 2, 3, 23, 24, 25, 47, 48, 49]]

# Regressor hyperparameters search space
def search_space(trial):
    search_space  = {
        'n_estimators'  : trial.suggest_int('n_estimators', 800, 1500, step=100), # number of decision trees
        'max_depth'     : trial.suggest_int('max_depth', 3, 10, step=1),          # max. depth of each  tree
        'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.5),        # step size towards the minimum of the loss function
        'reg_alpha'     : trial.suggest_float('reg_alpha', 0, 1, step=0.1),       # L1 regularization
        'reg_lambda'    : trial.suggest_float('reg_lambda', 0, 1, step=0.1),      # L2 regularization
    } 
    return search_space

# Perform the search
# ==============================================================================
results_search, frozen_trial = bayesian_search_forecaster(
                                   forecaster         = forecaster,
                                   y                  = data.loc[:, columnName],
                                   steps              = steps,
                                   metric             = 'mean_absolute_error',
                                   search_space       = search_space,
                                   lags_grid          = lags_grid,
                                   initial_train_size = len(data[:end_train]),
                                   refit              = False, 
                                   n_trials           = 100,    # Increase for more exhaustive search
                                   random_state       = 123,    # To reproduce results
                                   return_best        = True,   # To return best forecast parameters
                                   n_jobs             = 'auto', # '-1' to use all available cores
                                   verbose            = False,
                                   show_progress      = True
                               )

# Search results
# ==============================================================================
results_search.head(10)
 """

# Backtest final model on test data
# ==============================================================================
metric, predictions = backtesting_forecaster(
                          forecaster         = forecaster,
                          y                  = data.loc[:,columnName],
                          steps              = steps,
                          metric             = 'mean_absolute_error',
                          initial_train_size = len(data[:end_train]),
                          refit              = False,
                          n_jobs             = 'auto',
                          verbose            = False, # Change to True to see detailed information
                          show_progress      = True
                      )

print(f"Backtest error (MAE): {metric:.2f}")

""" # Plot predictions vs real value
# ======================================================================================
fig = go.Figure()
trace1 = go.Scatter(x=data_test.index, y=data_test[columnName], name="test", mode="lines")
trace2 = go.Scatter(x=predictions.index, y=predictions['pred'], name="prediction", mode="lines")
fig.add_trace(trace1)
fig.add_trace(trace2)
fig.update_layout(
    title="Real value vs predicted in test data",
    xaxis_title="Date time",
    yaxis_title="Demand",
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
fig.write_html('fig_after_hyper_param.html') """

params = {
    'n_estimators': 1300,
    'max_depth': 5,
    'learning_rate': 0.12115721224645952,
    'reg_alpha': 0.6000000000000001,
    'reg_lambda': 0.7000000000000001,
    'random_state': 15926,
    'verbose': -1
}
lags = 24

forecaster = ForecasterAutoreg(
                 regressor = LGBMRegressor(**params),
                 lags      = lags
             )

exog_features = ['generation fossil gas']

# Backtesting model
# ==============================================================================
metric, predictions = backtesting_forecaster(
                          forecaster         = forecaster,
                          y                  = data[columnName],
                          exog               = data[exog_features],
                          steps              = 24,
                          metric             = 'mean_absolute_error',
                          initial_train_size = len(data[:end_train]),
                          refit              = False,
                          n_jobs             = 'auto',
                          verbose            = False,
                          show_progress      = True
                      )

correlations = df_final.corr(method='pearson')
print(correlations[columnName].sort_values(ascending=False).to_string())

correlations = df_final.corr(method='pearson')
highly_correlated = abs(correlations[correlations > 0.5])
print(highly_correlated[highly_correlated < 1.0].stack().to_string())

# extract those that are higly correlated one to the other
index_values = highly_correlated.index[highly_correlated.index == columnName]
selected_features = highly_correlated.loc[:, index_values]
print(selected_features[selected_features < 1.0].stack().to_string())


""" # Assuming 'feature' is the column you want to normalize
feature_to_normalize = data['Load'].values.reshape(-1, 1)  # Reshape to 2D array for MinMaxScaler

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the feature data
normalized_feature = scaler.fit_transform(feature_to_normalize)

# Replace the original feature column with the normalized values
data['normalized_feature'] = normalized_feature """

st = 1