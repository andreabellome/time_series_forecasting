import pandas as pd
import plotly.graph_objects as go
from my_app.functions.processAndPlotData import processAndPlotData

instancePlotData = processAndPlotData()

combined_data = pd.read_csv('datasets/combined_data.csv')
data = pd.read_csv('datasets/load.csv')


dataProcessed = instancePlotData.process_data(data)
combined_dataProcessed = instancePlotData.process_data(combined_data)

fig1 = instancePlotData.plot_raw_data( combined_dataProcessed, 'PUN', 'Price', 'Time', 'Price' )
fig1.show()

""" fig2 = instancePlotData.plot_raw_data( dataProcessed, 'Load', 'Load', 'Time', 'Load' )
fig2.show() """

fig3 = instancePlotData.plot_distribution_by_month(combined_dataProcessed, 'PUN')
fig3.show()

fig4 = instancePlotData.plot_distribution_by_week(combined_dataProcessed, 'PUN')
fig4.show()

fig5 = instancePlotData.plot_distribution_by_years(combined_dataProcessed, 'PUN')
fig5.show()

st = 1

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