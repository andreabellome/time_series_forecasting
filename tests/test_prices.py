import pandas as pd
import plotly.graph_objects as go

# Specify the path to your Excel file
excel_file = "./datasets/Prices/Anno2017.xlsx"

# Specify the sheet name you want to load
sheet_name = "Prezzi-Prices"

# Load the Excel file into a DataFrame
df = pd.read_excel(excel_file, sheet_name=sheet_name)

# modify the data
df.columns.values[0] = "Date" # rename the first column
df.columns.values[1] = "Hour" # rename the second column
df = df[df['Hour'] != 25]     # avoid the 25 (?)
df['Hour'] -= 1               # shift from 0 to 23 hours (datetime does not work with 24)

# merge date and hour and convert to datetime
df['Date'] = pd.to_datetime(df['Date'].astype(str) + df['Hour'].astype(str).str.zfill(2), format='%Y%m%d%H')

# plot
fig = go.Figure()
trace1 = go.Scatter( x=df['Date'], y=df['PUN'], name="to predict", mode="lines")

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