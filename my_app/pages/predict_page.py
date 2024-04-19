import streamlit as st

@st.cache_data(experimental_allow_widgets=True)
def show_predict_page():

    st.title("Power-load Forecast Web-App :rocket:")
    st.subheader("Time-series forecasting using Machine Learning")

    n_days = st.slider("Days of prediction: ", 1, 365)
    
    st.warning("Page currently under construction... please come back later!")
