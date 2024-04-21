import streamlit as st
import pandas as pd
import numpy as np

from ..functions.plotData import plotData
import pickle
from plotly import graph_objects as go

@st.cache_data
def plot_predictions(predictions):
    # Plot predictions
    # ======================================================================================
    fig = go.Figure()
    trace2 = go.Scatter(x=predictions.index, y=predictions, name="prediction", mode="lines")
    fig.add_trace(trace2)
    fig.update_layout(
        title="Real value vs predicted in test data (Wh)",
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
        ),
        xaxis_rangeslider_visible=True
    )
    st.plotly_chart(fig)

@st.cache_data(experimental_allow_widgets=True)
def show_predict_page():

    st.title("Power-load Forecast Web-App :rocket:")
    st.subheader("Time-series forecasting using Machine Learning")
    
    # select the loads
    loads = ("Hospital 1", "ASL 1", "ASL 2")
    selected_loads = st.selectbox("Select dataset for prediction", loads)

    # initialize the data load and plot
    plotDataInstance = plotData()
    data, okay, message = plotDataInstance.load_data_and_process(selected_loads)

    n_days = st.slider("Days of prediction: ", 7, 365)
    nprevdays = n_days

    if okay:

        # Load the forecaster object from the file
        with open('forecaster_params.pkl', 'rb') as file:
            forecaster = pickle.load(file)

        if st.button("Generate forecast"):

            # Make predictions
            # ======================================================================================
            lags = forecaster.lags[-1]
            predictions = forecaster.predict(n_days, data.loc[:, 'Load'].iloc[-lags:])

            # Plot the predictions
            # ======================================================================================
            plot_predictions(predictions)

        else:

            # show a default image with 365 days of forecast horizon
            lags = forecaster.lags[-1]
            predictions = forecaster.predict(nprevdays, data.loc[:, 'Load'].iloc[-lags:])

            plot_predictions(predictions)
            
            

    else:
        st.warning(message)
