import streamlit as st
import pandas as pd
from app.const import (
    DATE_TIME_PATTERN,
    STREAMLIT_OUTPUT_FILE_NAME_PREFIX,
    RESULTS_FOLDER,
    STREAMLIT_INPUT_FILE_NAME_PREFIX,
    DATA_FOLDER,
    STREAMLIT_PREPROCESSED_FILE_NAME_PREFIX,
)
import subprocess as sp
from datetime import datetime
import os
import time
import random


# Session variables initialization
if "session_postfix" not in st.session_state:
    st.session_state.session_postfix = f"{datetime.now().strftime(DATE_TIME_PATTERN)}"

if "uploader_key_id" not in st.session_state:
    st.session_state.uploader_key_id = 0

if "file_uploader_disabled" not in st.session_state:
    st.session_state.file_uploader_disabled = False

if "make_predictions" not in st.session_state:
    st.session_state.make_predictions = False

if "predictions" not in st.session_state:
    st.session_state.predictions = None


# Callback functions definition
def reset_upload():
    st.session_state.session_postfix = f"{datetime.now().strftime(DATE_TIME_PATTERN)}"
    st.session_state.uploader_key_id += 1
    st.session_state.file_uploader_disabled = False
    st.session_state.make_predictions = False
    st.session_state.predictions = None


def handle_file_pick():
    if (
        st.session_state[f"file_uploader_state{st.session_state.uploader_key_id}"]
        is None
    ):
        # st.session_state.file_uploader_disabled = False
        reset_upload()
    else:
        st.session_state.file_uploader_disabled = True


def make_predictions():
    st.session_state.make_predictions = True


# Webpage structure and logic
st.title("CTR Predictor 🖱️📊")
st.header("Upload sessions data and get Click-Through Rate (CTR) predictions!")

# Input file upload section
st.subheader("📁 Upload your dataset (CSV format)")

uploaded_file = st.file_uploader(
    "Choose a file",
    type="csv",
    label_visibility="collapsed",
    disabled=st.session_state.file_uploader_disabled,
    on_change=handle_file_pick,
    key=f"file_uploader_state{st.session_state.uploader_key_id}",
)

if uploaded_file is not None:

    # If an input fil was selected
    # Read it
    df = pd.read_csv(uploaded_file)

    # Preview it
    st.subheader("👀 Preview of Uploaded Data")
    st.dataframe(df.head(), hide_index=True)

    if not st.session_state.make_predictions:

        # If the user has not chosen yet to make predictions
        # Show the user the available action options
        col1, col2, col3 = st.columns([5, 10, 2.25])
        with col3:
            st.button(
                "↻ Reset",
                on_click=reset_upload,
                disabled=st.session_state.make_predictions,
            )
        with col1:
            st.button(
                "🚀 Make predictions",
                on_click=make_predictions,
                disabled=st.session_state.make_predictions,
            )
    else:
        # If the user has chosen to make predictions
        # Set the output file name and path
        predictions_file_name = f"{STREAMLIT_OUTPUT_FILE_NAME_PREFIX}_{st.session_state.session_postfix}.csv"
        predictions_file_path = os.path.join(RESULTS_FOLDER, predictions_file_name)

        if st.session_state.predictions is None:

            # If predictions have not been made
            # Make predictions
            # Initialize progress bar
            progress_bar = st.progress(0, text="Preparing data file...")
            time.sleep(0.75)

            # Set the input file internal name and path
            raw_inference_file_path = os.path.join(
                DATA_FOLDER,
                f"{STREAMLIT_INPUT_FILE_NAME_PREFIX}_{st.session_state.session_postfix}.csv",
            )

            # Set the preprocessed file internal name and path
            preprocessed_inference_file_name = f"{STREAMLIT_PREPROCESSED_FILE_NAME_PREFIX}_{st.session_state.session_postfix}.csv"
            preprocessed_inference_file_path = os.path.join(
                DATA_FOLDER, preprocessed_inference_file_name
            )

            # Save the input file locally
            df.to_csv(
                raw_inference_file_path,
                index=False,
            )

            # Update the progress bar accordingly
            progress_bar.progress(1 / 3, text="Preprocessing data...")

            # Preprocess the input file
            cmd = f"python preprocess.py --inference-run True --csv-raw-path {raw_inference_file_path} --inference-output-file-name {preprocessed_inference_file_name}"
            result = sp.run(cmd, shell=True, check=True, capture_output=True, text=True)

            # Update the progress bar accordingly
            progress_bar.progress(2 / 3, text="Making predictions...")

            # Make predictions on the preprocessed file
            cmd = f"python predict.py --ignore-wandb --input-data-path {preprocessed_inference_file_path} --predictions-only True --predictions-file-name {predictions_file_name}"
            # result = sp.run(cmd, shell=True, check=True, capture_output=True, text=True)
            process = sp.Popen(
                cmd, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT, text=True
            )
            progress = 2 / 3
            while process.poll() is None:
                increment = random.uniform(0.005, 0.01)
                progress = min(progress + increment, 0.95)
                progress_bar.progress(progress, text="Making predictions...")
                time.sleep(random.uniform(0.8, 1.5))

            # Update the progress bar accordingly
            progress_bar.progress(1.0, text="Done!")
            time.sleep(1)
            progress_bar.empty()

        # Open the output file
        with open(predictions_file_path, "rb") as file:

            # Preview the predictions
            st.subheader("🔮 Preview of CTR Predictions")

            st.session_state.predictions = pd.read_csv(file, header = None)
            st.session_state.predictions.columns = ["predictions"]
            st.dataframe(
                st.session_state.predictions.head(),
                hide_index=True,
            )

            # Show the user the available action options
            col1, col2, col3 = st.columns([5, 10, 5])
            with col3:
                st.button("✨ Make New Predictions", on_click=reset_upload)
            with col1:
                st.download_button(
                    label="📥 Download Predictions",
                    data=file,
                    file_name="predictions.csv",
                    mime="text/csv",
                )
