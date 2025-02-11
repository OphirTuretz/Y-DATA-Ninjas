import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from invoke import Context
from app.const import DEFAULT_STREAMLIT_INFERENCE_PATH, DEFAULT_CSV_PREDICTIONS_INFERENCE_PATH

from tasks import streamlit_pipe

st.title('Y-DATA-Ninjas Inference Point')

uploaded_file = st.file_uploader('Upload your dataset here:')

if uploaded_file:
    st.header('Data Stats:')
    df=pd.read_csv(uploaded_file)
    df.to_csv(DEFAULT_STREAMLIT_INFERENCE_PATH)
    c=Context()
    streamlit_pipe(c)
    df_inf = pd.read_csv(DEFAULT_CSV_PREDICTIONS_INFERENCE_PATH)
    st.download_button("Download resulting dataframe as CSV", df_inf.to_csv(index=False), "results.csv")