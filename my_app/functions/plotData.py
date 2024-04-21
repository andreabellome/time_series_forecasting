import streamlit as st
import os
import pandas as pd
import numpy as np
from plotly import graph_objects as go
import matplotlib.pyplot as plt

class plotData:

    # Dictionary mapping loads to corresponding CSV files
    load_csv_mapping = {
        "Hospital 1": "load.csv",
        "ASL 1": "asl_whatever.csv",
        "ASL 2": "asl_another.csv"
    }

    def __init__(self):
        pass
    
    @st.cache_data
    def load_data_and_process(_self, ticker):
        """
        Function to load and process data based on the selected ticker.
        
        Args:
            ticker (str): The selected ticker from the dropdown.
        
        Returns:
            data (DataFrame): The processed DataFrame.
            okay (bool): Whether the operation was successful.
            message (str): Message indicating the outcome of the operation.
        """
        
        csv_filename = _self.load_csv_mapping[ticker]
        try:
            data = pd.read_csv(os.path.join("datasets", csv_filename))
            data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
            data = data.set_index('Date')
            data = data.asfreq('1D')
            data = data.sort_index()
            if data.isnull().any(axis=1).mean() > 0.0:
                data.asfreq(freq='1D', fill_value=np.nan)
            message = "Everything okay."
            return data, True, message
        except FileNotFoundError:
            message = "File not found, please check the database."
            return np.NaN, False, message
        

    @st.cache_data
    def plot_raw_data(_self, data):
        """
        Function to plot the raw data.
        
        Args:
            data (DataFrame): The DataFrame containing the data to be plotted.
        """
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
            xaxis_rangeslider_visible=True
        )

        return fig

    @st.cache_data
    def plot_zoom(_self, data, start_date, end_date):
        """
        Function to plot the zoomed-in data.
        
        Args:
            data (DataFrame): The DataFrame containing the data to be plotted.
            start_date (datetime): The start date for zooming.
            end_date (datetime): The end date for zooming.
        """
        fig = plt.figure(figsize=(10, 5))
        if start_date >= end_date:
            st.warning("Please, select a start date that is greater than the end date to analyse.")
            return False
        else:
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

            return fig
    
    @st.cache_data
    def plot_distribution_by_month(_self, data):
        # Load distribution by month
        # ==============================================================================
        
        # Compute month
        data['month'] = data.index.month

        # Compute median values for each month
        median_values = data.groupby('month')['Load'].median()

        # Create a box plot
        box_trace = go.Box(
            x=data['month'],
            y=data['Load'],
            boxmean='sd',
            name='Load',
            marker_color='skyblue'
        )

        # Add median lines
        median_trace = go.Scatter(
            x=median_values.index,
            y=median_values.values,
            mode='lines+markers',
            name='Median',
            line=dict(color='red', width=1)
        )

        # Layout settings
        layout = go.Layout(
            title='Load distribution by month (Wh)',
            xaxis=dict(title='Month'),
            yaxis=dict(title='Demand'),
        )

        # Combine traces and layout
        fig = go.Figure(data=[box_trace, median_trace], layout=layout)

        return fig
    
    @st.cache_data
    def plot_distribution_by_week(_self, data):
        # Load distribution by month
        # ==============================================================================

        # Compute week day
        data['week_day'] = data.index.dayofweek + 1

        # Compute median values for each week day
        median_values = data.groupby('week_day')['Load'].median()

        # Create a box plot
        box_trace = go.Box(
            x=data['week_day'],
            y=data['Load'],
            boxmean='sd',
            name='Load',
            marker_color='skyblue'
        )

        # Add median lines
        median_trace = go.Scatter(
            x=median_values.index,
            y=median_values.values,
            mode='lines+markers',
            name='Median',
            line=dict(color='red', width=1)
        )

        # Layout settings
        layout = go.Layout(
            title='Load distribution by week day (Wh)',
            xaxis=dict(title='Week Day'),
            yaxis=dict(title='Demand'),
        )

        # Combine traces and layout
        fig = go.Figure(data=[box_trace, median_trace], layout=layout)

        return fig
    
    @st.cache_data
    def plot_distribution_by_years(_self, data):
        # Load distribution by month
        # ==============================================================================
        
        # Compute year
        data['year'] = data.index.year

        # Compute median values for each year
        median_values = data.groupby('year')['Load'].median()

        # Create a box plot
        box_trace = go.Box(
            x=data['year'],
            y=data['Load'],
            boxmean='sd',
            name='Load',
            marker_color='skyblue'
        )

        # Add median lines
        median_trace = go.Scatter(
            x=median_values.index,
            y=median_values.values,
            mode='lines+markers',
            name='Median',
            line=dict(color='red', width=1)
        )

        # Layout settings
        layout = go.Layout(
            title='Load distribution by year (Wh)',
            xaxis=dict(title='Year'),
            yaxis=dict(title='Demand'),
        )

        # Combine traces and layout
        fig = go.Figure(data=[box_trace, median_trace], layout=layout)

        return fig
        

