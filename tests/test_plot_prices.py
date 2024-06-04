import pandas as pd
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

combined_data = pd.read_csv('datasets/combined_data.csv') # these are the data of the PUN 
data = pd.read_csv('datasets/load.csv')
data_temperature = pd.read_csv('datasets/italy_temperatures.csv')

""" data_temperature = pd.read_csv('datasets/city_temperature.csv')

data_temperature = data_temperature[data_temperature["Region"] == "Europe"]
data_temperature = data_temperature[data_temperature["Country"] == "Italy"]

data_temperature = data_temperature[data_temperature['Year'] >= 1000]
data_temperature = data_temperature[data_temperature['Day'] >= 1]
data_temperature = data_temperature[data_temperature['Month'] >= 1]

data_temperature = data_temperature[data_temperature["City"] == "Rome"]
data_temperature = data_temperature[data_temperature["Year"] >= 2006]
data_temperature = data_temperature[data_temperature["Year"] <= 2019]

data_temperature['Date'] = pd.to_datetime(data_temperature[['Year', 'Month', 'Day']])
data_temperature.drop(columns=['Year', 'Month', 'Day'], inplace=True)
data_temperature = data_temperature.set_index('Date')
data_temperature = data_temperature[~data_temperature.index.duplicated()]

data_temperature = data_temperature.asfreq('1D')
data_temperature = data_temperature.sort_index() """

data_temperature['AvgTemperature'] = data_temperature['AvgTemperature'].map(lambda x: (x - 32) * (5/9))

data_temperature = data_temperature[data_temperature["City"] == "Rome"]
data_temperature = data_temperature[data_temperature["Year"] >= 2006]
# data_temperature = data_temperature[data_temperature["Year"] <= 2019]

dataProcessed = instancePlotData.process_data(data)
combined_dataProcessed = instancePlotData.process_data(combined_data)
data_temperature = instancePlotData.process_data(data_temperature)

fig1 = instancePlotData.plot_raw_data( combined_dataProcessed, 'PUN', 'Price', 'Time', 'Price' )
fig1.show()

fig2 = instancePlotData.plot_raw_data( data_temperature, 'AvgTemperature', 'Temp', 'Time', 'Temperature' )
fig2.show()
fig2.write_html('temperature_rome.html')

""" fig2 = instancePlotData.plot_raw_data( dataProcessed, 'Load', 'Load', 'Time', 'Load' )
fig2.show() """

""" fig3 = instancePlotData.plot_distribution_by_month(combined_dataProcessed, 'PUN')
fig3.show()

fig4 = instancePlotData.plot_distribution_by_week(combined_dataProcessed, 'PUN')
fig4.show()

fig5 = instancePlotData.plot_distribution_by_years(combined_dataProcessed, 'PUN')
fig5.show() """

st = 1

combined_data.PUN.plot(c='black', alpha=0.5, linewidth=0.5)


# plot
fig = go.Figure()
trace1 = go.Scatter( x=combined_data['Date'], y=combined_data['PUN'], name="to predict", mode="lines")

fig.add_trace(trace1)

fig.update_layout(
    title="Electrical prices",
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

st = 1