import pandas as pd
import plotly.graph_objects as go

# Specify the path to your Excel file
excel_file = "./datasets/Prices/Anno2017.xlsx"

# Specify the sheet name you want to load
sheet_name = "Prezzi-Prices"

# Load the Excel file into a DataFrame
df = pd.read_excel(excel_file, sheet_name=sheet_name)

df.columns.values[0] = "Date"
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

fig = go.Figure()
trace1 = go.Scatter( y=df['PUN'], name="to predict", mode="lines")

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