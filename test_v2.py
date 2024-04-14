# Data manipulation
# ==============================================================================
import numpy as np
import pandas as pd

# Plots
# ==============================================================================
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.offline as poff

pio.renderers.default = 'notebook' 
pio.templates.default = "seaborn"
poff.init_notebook_mode(connected=True)
plt.style.use('seaborn-v0_8-darkgrid')

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
data = pd.read_csv('datasets/italian-power-load/load.csv')

# Data preparation
# ==============================================================================
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
data = data.set_index('Date')
data = data.asfreq('1D')
data = data.sort_index()

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
end_test = '2019-12-31'

train = data.loc[:end_train, :].copy()
test = data.loc[start_test:, :].copy()

sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(2, 1, 1, 365))
sarima_model_fit = sarima_model.fit(disp=0)
print(sarima_model_fit.summary())

sarima_predictions = sarima_model_fit.predict(start=start_test, end=end_test)

# Figure
fig, ax = plt.subplots(figsize=(16,9), facecolor='w')

ax.plot(test, label='Testing Set')
ax.plot(sarima_predictions, label='Forecast')

# Labels
ax.set_title("Test vs Predictions", fontsize=15, pad=10)
ax.set_ylabel("Number of orders", fontsize=12)
ax.set_xlabel("Date", fontsize=12)

# Legend & Grid
ax.grid(linestyle=":", color='grey')
ax.legend()

plt.show()

st = 1

