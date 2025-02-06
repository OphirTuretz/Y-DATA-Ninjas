import streamlit as st
import subprocess
import pickle
import pandas as pd

DEFAULT_MODEL_PATH = "archived_experiments/2025-02-06_19-35-52_experiment-nql66hwu/models/model.pkl"
def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model
    

st.title("CTR Ninjas magnificent model")
st.write("Welcome to the platform that will help you predict the click-through rate of your ads.")
data = st.file_uploader("Upload your data", type=["csv"])
if data:
    data = pd.read_csv(data)
    st.write("Data uploaded successfully, have a peek:")
    st.dataframe(data)
    st.write("Now you can proceed with the prediction.")
    if st.button("Predict"):
        st.write("Prediction is in progress...")
        
        data.to_csv("data/inference.csv", index=False)
        st.write("Don't worry about the data, it's saved.")
        model = load_model(DEFAULT_MODEL_PATH)

        # Currently overriding inference.csv
        subprocess.run(["python", "preprocess.py", "-i", "data/inference.csv", "-infr", "True"])

        # Using default model, should be best model from wandb
        subprocess.run(["python", "predict.py", "-i", "data/inference.csv",
                                       "-m", DEFAULT_MODEL_PATH, '-o', "inference.csv", '-po', 'True'])
        st.write("Congrats friend, predictions are made, you can download them below.")
        st.write("Download your predictions here:")

        # preview the predictions
        predictions = pd.read_csv("results/inference.csv")
        st.dataframe(predictions)
        if st.download_button(label="Download predictions", data="results/inference.csv", file_name="streamlit_predictions.csv"):
            st.write("Downloaded successfully.")


