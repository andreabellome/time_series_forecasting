import pandas as pd
import plotly.graph_objects as go
import os

wantToLoadAgain = False

if wantToLoadAgain:

    # path where you find the prices
    folder_path = "./datasets/Prices"
    folder_contents = os.listdir(folder_path)

    # Specify the sheet name you want to load
    sheet_name = "Prezzi-Prices"

    # scrape all the folders to check the data in them
    all_data = []
    for folder in folder_contents:

        inner_folder = folder_path + "/" + folder + "/"
        files = [f for f in os.listdir(inner_folder) if os.path.isfile(os.path.join(inner_folder, f))]
        load_file = inner_folder + files[0]

        # load the data for the given year
        df = pd.read_excel(load_file, sheet_name=sheet_name)

        # modify the data
        df.columns.values[0] = "Date" # rename the first column
        df.columns.values[1] = "Hour" # rename the second column
        df = df[df['Hour'] != 25]     # avoid the 25 (?)
        df['Hour'] -= 1               # shift from 0 to 23 hours (datetime does not work with 24)

        # merge date and hour and convert to datetime
        df['Date'] = pd.to_datetime(df['Date'].astype(str) + df['Hour'].astype(str).str.zfill(2), format='%Y%m%d%H')
        df.drop(columns=['Hour'], inplace=True)

        # creation of the dataset
        data = df.groupby(df['Date'].dt.date)[df.columns[1:]].mean().reset_index() # take the mean for each value

        # append the data
        all_data.append(data)

        print("Loaded file: " + load_file)

    combined_data = pd.concat(all_data, ignore_index=True)          # concatenate all the data
    combined_data.to_csv('datasets/combined_data.csv', index=False) # save all the data to file


combined_data = pd.read_csv('datasets/combined_data.csv') # these are the data of the PUN 
data = pd.read_csv('datasets/load.csv')                   # these are the data of the power load

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