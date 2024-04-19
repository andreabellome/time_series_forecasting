import streamlit as st

import pandas as pd
import numpy as np

import os

from plotly import graph_objects as go
import matplotlib.pyplot as plt

# Dictionary mapping loads to corresponding CSV files
load_csv_mapping = {
    "Hospital 1": "load.csv",
    "ASL 1": "asl_whatever.csv",
    "ASL 2": "asl_another.csv"
}

@st.cache_data(experimental_allow_widgets=True)
def show_data_analysis_page():

    @st.cache_data
    def load_data_and_process(ticker):

        # Extract the CSV filename from the dictionary
        # ==============================================================================
        csv_filename = load_csv_mapping[ticker]

        try:
            # Loading the data
            # ==============================================================================
            data = pd.read_csv(os.path.join("datasets", csv_filename))
            
            # Data preparation --> convert to datetime and set the index
            # ==============================================================================
            data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
            data = data.set_index('Date')
            data = data.asfreq('1D')
            data = data.sort_index()

            # Verify that the time series is complete, if not fill with nan values
            # ==============================================================================
            (data.index == pd.date_range(start=data.index.min(),
                                        end=data.index.max(),
                                        freq=data.index.freq)).all()

            # if not complete, fill with NaN values
            if data.isnull().any(axis=1).mean() > 0.0:
                data.asfreq(freq='1D', fill_value=np.nan)

            message = "Everything okay."
            return data, True, message
        except FileNotFoundError:
            message = "File not found, please check the database."
            return np.NaN, False, message

    @st.cache_data
    def plot_raw_data(data):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Load'], mode='lines', name='Load'))
        fig.update_layout(
            title  = 'Power load (Wh)',
            xaxis_title="Time",
            yaxis_title="Load",
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
            ),
            xaxis_rangeslider_visible = True
        )
        st.plotly_chart(fig)

    @st.cache_data
    def plot_zoom(data, start_date, end_date):

        if start_date >= end_date:
            st.warning("Please, select a start date that is greater than the end date to analyse.")
        else:

            # Zooming time series chart
            # ==============================================================================
            start_date = start_date.strftime("%Y-%m-%d")
            end_date = end_date.strftime("%Y-%m-%d")
            zoom = (start_date, end_date)

            fig = plt.figure(figsize=(10, 5))
            grid = plt.GridSpec(nrows=8, ncols=1, hspace=0.6, wspace=0)

            main_ax = fig.add_subplot(grid[:3, :])
            data.Load.plot(ax=main_ax, c='black', alpha=0.5, linewidth=0.5)
            min_y = min(data.Load)
            max_y = max(data.Load)
            main_ax.fill_between(zoom, min_y, max_y, facecolor='blue', alpha=0.5, zorder=0)
            main_ax.set_title(f'Power load (Wh): {data.index.min()}, {data.index.max()}', fontsize=10)
            main_ax.set_xlabel('')

            zoom_ax = fig.add_subplot(grid[5:, :])
            data.loc[zoom[0]: zoom[1]].Load.plot(ax=zoom_ax, color='blue', linewidth=1)
            zoom_ax.set_title(f'Power load (Wh): {zoom}', fontsize=10)
            zoom_ax.set_xlabel('')

            st.pyplot(fig)

    # Web-app starts here
    # ==============================================================================

    st.title("Power-load Forecast Web-App :rocket:")
    st.subheader("Data analysis section")

    loads = ( "Hospital 1", "ASL 1", "ASL 2" )
    selected_loads = st.selectbox("Select dataset for prediction", loads)

    n_days = st.slider("Days of prediction: ", 1, 365)

    data, okay, message = load_data_and_process(selected_loads)

    if okay:
        plot_raw_data(data)

        defaultStartZoom = '2018-01-01'
        defaultEndZoom = '2018-12-31'
        dataZoom = data.loc[defaultStartZoom:defaultEndZoom, :].copy()


        # Get start and end dates from the user
        start_date = st.date_input("Select start date:", min_value=data.index.min(), max_value=data.index.max(), value=dataZoom.index.min())
        end_date = st.date_input("Select end date:", min_value=data.index.min(), max_value=data.index.max(), value=dataZoom.index.max())

        # Generate plot if the user clicks the button
        if st.button("Generate Plot"):
            plot_zoom(data, start_date, end_date)

    else:
        st.warning(message)

