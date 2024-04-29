import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as poff
import matplotlib.pyplot as plt
from my_app.functions.processAndPlotData import processAndPlotData

pio.renderers.default = "browser"
pio.templates.default = "seaborn"
poff.init_notebook_mode(connected=True)
plt.style.use('seaborn-v0_8-darkgrid')

instancePlotData = processAndPlotData()

dataTemperature = pd.read_csv('datasets/weather_features.csv')
df_energy = pd.read_csv('datasets/energy_dataset.csv')

dataTemperature.head(10)
dataTemperature.info()

df_energy = df_energy.drop(['generation fossil coal-derived gas',
                           'generation fossil oil shale',
                           'generation fossil peat', 
                            'generation geothermal',
            'generation hydro pumped storage aggregated', 
            'generation marine', 'generation wind offshore',
        'forecast wind offshore eday ahead', 'total load forecast',
        'forecast solar day ahead', 'forecast wind onshore day ahead'],
                          axis=1)

df_energy['time'] = pd.to_datetime(df_energy['time'],
                                  utc=True, 
                            infer_datetime_format = True)
df_energy = df_energy.set_index('time')

df_energy.interpolate(method='linear', limit_direction='forward',
                inplace=True, axis=0)

fig1 = instancePlotData.plot_raw_data( df_energy, 'total load actual', 'Load', 'Time', 'Load [MWh]', usingPlt=True )
fig1.show()

st = 1
