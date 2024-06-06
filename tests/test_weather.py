import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

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
df_weather = pd.read_csv('datasets/weather_features.csv')


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
# fig1.show()

df_weather = instancePlotData.df_convert_dtypes(df_weather, np.int64, np.float64)

df_weather['time'] = pd.to_datetime(df_weather['dt_iso'], utc=True,
                                   infer_datetime_format=True)
df_weather = df_weather.drop(['dt_iso'], axis=1)
df_weather = df_weather.set_index('time')

print('There are {} missing values or NaNs in df_weather.'.format(df_weather.isnull().values.sum()))

temp_weather = df_weather.duplicated(keep='first').sum()

print('There are {} duplicate rows in df_weather based on all columns.'.format(temp_weather))


# Display the numebr of rows in each dataframe
print('There are {} observations in df_energy.'.format(df_energy.shape[0]))

cities = df_weather['city_name'].unique()
grouped_weather = df_weather.groupby('city_name')

for city in cities:
    print('There are {} observations in df_weather'.format(grouped_weather.get_group('{}'.format(city)).shape[0]),
         'about city: {}.'.format(city))

# Create df_weather_2 and drop duplicate rows in df_weather
df_weather_2 = df_weather.reset_index().drop_duplicates(subset=
                                    ['time', 'city_name'],
                            keep='last').set_index('time')

df_weather = df_weather.reset_index().drop_duplicates(subset=[
    'time', 'city_name'
], keep='first').set_index('time')

# Display the number of rows in each dataframe again
print('There are {} observations in df_energy.'.format(df_energy.shape[0]))

grouped_weather = df_weather.groupby('city_name')

for city in cities:
    print('There are {} observations in df_weather'
          .format(grouped_weather.get_group('{}'.format(city)).shape[0]), 
          'about city: {}.'.format(city))

# Display all the unique values in teh column 'weather_description'
weather_description_unique = df_weather['weather_description'].unique()
weather_description_unique

# Display all the unique values in the column 'weather_id'
weather_id_unique = df_weather['weather_id'].unique()
weather_id_unique

# Define a function which will calculate R-squared score for the same column in two datframes
def encode_and_display_r2_score(df_1, df_2, column, categorical=False):
    dfs = [df_1, df_2]
    if categorical:
        for df in dfs:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
    r2 = r2_score(df_1[column], df_2[column])
    print("R-Squared score of {} is {}".format(column, r2.round(3)))


# Display the R-squared scores for the columns with qualitative weather descriptions in df_weather and df_weather_2
encode_and_display_r2_score(df_weather, df_weather_2, 'weather_description', categorical=True)
encode_and_display_r2_score(df_weather, df_weather_2, 'weather_main', categorical=True)
encode_and_display_r2_score(df_weather, df_weather_2, 'weather_id')

# Drop columns with qualitative weather information
df_weather = df_weather.drop(['weather_main', 'weather_id',
                'weather_description', 'weather_icon'],
                            axis=1)


# Display the R-squared for all the columns in df_weather and df_weather_2
df_weather_cols = df_weather.columns.drop('city_name')
for col in df_weather_cols:
    encode_and_display_r2_score(df_weather, df_weather_2, col)


# Display the number of duplicates in df_weather
temp_weather = df_weather.reset_index().duplicated(subset=['time', 'city_name'], 
                                                   keep='first').sum()
print('There are {} duplicate rows in df_weather ' \
      'based on all columns except "time" and "city_name".'.format(temp_weather))


# Replace outliers in 'pressure' with NaNs
df_weather.loc[df_weather.pressure > 1051, 'pressure'] = np.nan
df_weather.loc[df_weather.pressure < 931, 'pressure'] = np.nan

# Replace outliers in 'wind_speed' with NaNs
df_weather.loc[df_weather.wind_speed > 50, 'wind_speed'] = np.nan

# Fill null values using interpolation
df_weather.interpolate(method='linear', limit_direction='forward', inplace=True, axis=0)

# Split the df_weather into 5 dataframes (one for each city)
df_1, df_2, df_3, df_4, df_5 = [x for _, x in df_weather.groupby('city_name')]
dfs = [df_1, df_2, df_3, df_4, df_5]

# Merge all dataframes into the final dataframe
df_final = df_energy

for df in dfs:
    city = df['city_name'].unique()
    city_str = str(city).replace("'", "").replace('[', '').replace(']', '').replace(' ', '')
    df = df.add_suffix('_{}'.format(city_str))
    df_final = df_final.merge(df, on=['time'], how='outer')
    df_final = df_final.drop('city_name_{}'.format(city_str), axis=1)
    
df_final.columns

# Display the number of NaNs and duplicates in the final dataframe
print('There are {} missing values or NaNs in df_final.'
      .format(df_final.isnull().values.sum()))

temp_final = df_final.duplicated(keep='first').sum()
print('\nThere are {} duplicate rows in df_energy based on all columns.'
      .format(temp_final))

# save to csv
df_final.to_csv('df_final_electricity_price.csv', index=True)

st = 1
