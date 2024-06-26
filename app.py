import streamlit as st
from my_app.pages import data_analysis_page
from my_app.pages import predict_page
import streamlit as st

dark = '''
<style>
    .stApp {
    background-color: black;
    }
</style>
'''

st.markdown(dark, unsafe_allow_html=True)


# Display selectbox in sidebar
selected_page = st.sidebar.selectbox("Data analysis or predict", ("Data analysis", "Predict"))

# Show data analysis page if selected
if selected_page == "Data analysis":
    data_analysis_page.show_data_analysis_page()
elif selected_page == "Predict":
    predict_page.show_predict_page()