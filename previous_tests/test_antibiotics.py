# Data manipulation
# ==============================================================================
import numpy as np
import pandas as pd
from astral.sun import sun
from astral import LocationInfo
from skforecast.datasets import fetch_dataset

# Plots
# ==============================================================================
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as poff
import plotly.offline as pyo

pio.renderers.default = 'notebook' 
pio.templates.default = "seaborn"
poff.init_notebook_mode(connected=True)
plt.style.use('seaborn-v0_8-darkgrid')

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

# Split the remaining data into train-validation-test (70-15-15)s
# ==============================================================================
number_of_dates = data.shape[0]
index_train = round(70/100 * number_of_dates)
index_validation = index_train + round(15/100 * number_of_dates)

start_train = data.index[0]
end_train = data.index[index_train - 1]

start_validation = data.index[index_train]
end_validation = data.index[index_validation-1]

start_test = data.index[index_validation]

print("Start date:", start_train)
print("End date:", end_train)
print("Start date:", start_validation)
print("End date:", end_validation)
print("Start date:", start_test)

data_train = data.loc[start_train:end_train, :].copy()
data_val   = data.loc[start_validation:end_validation, :].copy()
data_test  = data.loc[start_test:, :].copy()

print(f"Train dates      : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
print(f"Validation dates : {data_val.index.min()} --- {data_val.index.max()}  (n={len(data_val)})")
print(f"Test dates       : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")

# Interactive plot of time series
# ==============================================================================
fig = go.Figure()
fig.add_trace(go.Scatter(x=data_train.index, y=data_train['R06'], mode='lines', name='Train'))
fig.add_trace(go.Scatter(x=data_val.index, y=data_val['R06'], mode='lines', name='Validation'))
fig.add_trace(go.Scatter(x=data_test.index, y=data_test['R06'], mode='lines', name='Test'))
fig.update_layout(
    title  = 'Sales of antihistamines R06',
    xaxis_title="Time",
    yaxis_title="R06",
    legend_title="Partition:",
    width=850,
    height=400,
    margin=dict(l=20, r=20, t=35, b=20),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1,
        xanchor="left",
        x=0.001
    )
)
fig.show()

# Zooming time series chart
# ==============================================================================
zoom = ('2017-01-01','2017-12-31')

fig = plt.figure(figsize=(10, 5))
grid = plt.GridSpec(nrows=8, ncols=1, hspace=0.6, wspace=0)

main_ax = fig.add_subplot(grid[:3, :])
data.R06.plot(ax=main_ax, c='black', alpha=0.5, linewidth=0.5)
min_y = min(data.R06)
max_y = max(data.R06)
main_ax.fill_between(zoom, min_y, max_y, facecolor='blue', alpha=0.5, zorder=0)
main_ax.set_title(f'Power R06 (Wh): {data.index.min()}, {data.index.max()}', fontsize=10)
main_ax.set_xlabel('')

zoom_ax = fig.add_subplot(grid[5:, :])
data.loc[zoom[0]: zoom[1]].R06.plot(ax=zoom_ax, color='blue', linewidth=1)
zoom_ax.set_title(f'Power R06 (Wh): {zoom}', fontsize=10)
zoom_ax.set_xlabel('')

plt.subplots_adjust(hspace=1)
plt.show()

# Load distribution by month
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 4))
data['month'] = data.index.month
data.boxplot(column='R06', by='month', ax=ax,)
data.groupby('month')['R06'].median().plot(style='o-', linewidth=0.8, ax=ax)
ax.set_ylabel('R06')
ax.set_title('R06 distribution by month (Wh)')
fig.suptitle('')
plt.show()

# Demand distribution by week day
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
data['week_day'] = data.index.day_of_week + 1
data.boxplot(column='R06', by='week_day', ax=ax)
data.groupby('week_day')['R06'].median().plot(style='o-', linewidth=0.8, ax=ax)
ax.set_ylabel('Demand')
ax.set_title('R06 distribution by week day (Wh)')
fig.suptitle('')
plt.show()

st = 1