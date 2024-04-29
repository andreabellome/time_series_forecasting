import pandas as pd
import numpy as np
from plotly import graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime

class processAndPlotData:

    def __init__(self):
        pass

    @staticmethod
    def process_data(data):
        data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
        data = data.set_index('Date')
        data = data[~data.index.duplicated()]
        data = data.asfreq('1D')
        data = data.sort_index()
        if data.isnull().any(axis=1).mean() > 0.0:
            data.asfreq(freq='1D', fill_value=np.nan)
        return data
    
    @staticmethod
    def plot_raw_data(data, columnName: str, 
                  title : str = 'Power load (Wh)', 
                  xaxis_title: str = "Time", 
                  yaxis_title: str ="Load", 
                  legend_title: str =None,
                  usingPlt: bool = False):
        """
        Function to plot the raw data.
        
        Args:
            data (DataFrame): The DataFrame containing the data to be plotted.
        """

        if usingPlt: 

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_xlabel(xaxis_title, fontsize=16)

            ax.plot(data[columnName], label=legend_title)
            ax.set_ylabel(yaxis_title, fontsize=16)
            ax.legend(fontsize=16)
            ax.set_title(title, fontsize=24)
            ax.grid(True)

            return fig
        
        else:

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data[columnName], mode='lines', name=columnName))
            fig.update_layout(
                title = title,  # Set title property
                xaxis_title = xaxis_title,  # Set xaxis_title property
                yaxis_title = yaxis_title,  # Set yaxis_title property
                legend_title = legend_title,  # Set legend_title property
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
            )

            return fig

            
    
    @staticmethod
    def plot_distribution_by_month(data, columnName: str):
        # Load distribution by month
        # ==============================================================================
        
        # Compute month
        data['month'] = data.index.month

        # Compute median values for each month
        median_values = data.groupby('month')[columnName].median()

        # Create a box plot
        box_trace = go.Box(
            x=data['month'],
            y=data[columnName],
            boxmean='sd',
            name=columnName,
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
    

    @staticmethod
    def plot_distribution_by_week(data, columnName: str):
        # Load distribution by month
        # ==============================================================================

        # Compute week day
        data['week_day'] = data.index.dayofweek + 1

        # Compute median values for each week day
        median_values = data.groupby('week_day')[columnName].median()

        # Create a box plot
        box_trace = go.Box(
            x=data['week_day'],
            y=data[columnName],
            boxmean='sd',
            name=columnName,
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
    
    @staticmethod
    def plot_distribution_by_years(data, columnName: str):
        # Load distribution by month
        # ==============================================================================
        
        # Compute year
        data['year'] = data.index.year

        # Compute median values for each year
        median_values = data.groupby('year')[columnName].median()

        # Create a box plot
        box_trace = go.Box(
            x=data['year'],
            y=data[columnName],
            boxmean='sd',
            name=columnName,
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