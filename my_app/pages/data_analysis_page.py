import streamlit as st
from datetime import datetime

from ..functions.plotData import plotData

# Dictionary mapping loads to corresponding CSV files
load_csv_mapping = {
    "Hospital 1": "load.csv",
    "ASL 1": "asl_whatever.csv",
    "ASL 2": "asl_another.csv"
}

dark = '''
<style>
    .stApp {
    background-color: black;
    }
</style>
'''

st.markdown(dark, unsafe_allow_html=True)

def show_data_analysis_page():

    # Web-app starts here
    st.title("Power-load Forecast Web-App :rocket:")
    st.subheader("Data analysis section")

    # select the loads
    loads = ("Hospital 1", "ASL 1", "ASL 2")
    selected_loads = st.selectbox("Select dataset for prediction", loads)
        
    # initialize the data load and plot
    plotDataInstance = plotData()
    data, okay, message = plotDataInstance.load_data_and_process(selected_loads)

    columnName = 'Load'
    if okay:
        
        # plot the entire dataset
        fig = plotDataInstance.plot_raw_data(data, columnName)
        st.plotly_chart(fig)

        # make a zoom on user-defined dates
        defaultStartZoom = '2018-01-01'
        defaultEndZoom = '2018-12-31'
        dataZoom = data.loc[defaultStartZoom:defaultEndZoom, :].copy()
        start_date = st.date_input("Select start date:", min_value=data.index.min(), max_value=data.index.max(), value=dataZoom.index.min())
        end_date = st.date_input("Select end date:", min_value=data.index.min(), max_value=data.index.max(), value=dataZoom.index.max())

        if st.button("Generate Plot"):
            fig = plotDataInstance.plot_zoom(data, columnName, start_date, end_date)
            if fig:
                st.pyplot(fig)
        else:
            fig = plotDataInstance.plot_zoom(data, columnName, start_date, end_date)
            if fig:
                st.pyplot(fig)

        
        # plot the load distribution by week
        wkeek_month_years = st.selectbox("Group by:", ("Week", "Month", "Year"), key=f"selectbox_1")

        if wkeek_month_years == "Week":
            stop = 1
            # plot the load distributions by months
            fig1 = plotDataInstance.plot_distribution_by_week(data, columnName)
            st.plotly_chart(fig1)
        elif wkeek_month_years == "Month":
            # plot the load distributions by months
            fig1 = plotDataInstance.plot_distribution_by_month(data, columnName)
            st.plotly_chart(fig1)
        else:
            # plot the load distributions by years
            fig1 = plotDataInstance.plot_distribution_by_years(data, columnName)
            st.plotly_chart(fig1)




    else:
        st.warning(message)
    
    
