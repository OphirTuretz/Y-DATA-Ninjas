import streamlit as st
import pandas as pd
from app.const import (
    DEFAULT_CSV_INFERENCE_PATH,
    DEFAULT_CSV_RAW_INFERENCE_PATH,
    CSV_PREDICTIONS_INFERENCE_FILENAME,
    DATE_TIME_PATTERN,
)
import subprocess as sp
from datetime import datetime
import os
import time
import random

INPUT_FILE_NAME = "inference"
OUTPUT_FILE_NAME = "predictions"

if "uploader_key_id" not in st.session_state:
    st.session_state.uploader_key_id = 0

if "file_uploader_disabled" not in st.session_state:
    st.session_state.file_uploader_disabled = False

if "session_postfix" not in st.session_state:
    st.session_state.session_postfix = f"{datetime.now().strftime(DATE_TIME_PATTERN)}"

if "make_predictions" not in st.session_state:
    st.session_state.make_predictions = False

if "predictions" not in st.session_state:
    st.session_state.predictions = None


def handle_file_pick():
    if (
        st.session_state[f"file_uploader_state{st.session_state.uploader_key_id}"]
        is None
    ):
        st.session_state.file_uploader_disabled = False
    else:
        st.session_state.file_uploader_disabled = True


def reset_upload():
    st.session_state.uploader_key_id += 1
    st.session_state.file_uploader_disabled = False
    st.session_state.make_predictions = False
    st.session_state.session_postfix = f"{datetime.now().strftime(DATE_TIME_PATTERN)}"
    st.session_state.predictions = None


def make_predictions():
    st.session_state.make_predictions = True


st.title("CTR Predictor 🖱️📊")
st.header("Upload sessions data and get Click-Through Rate (CTR) predictions!")

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
    df = pd.read_csv(uploaded_file)

    st.subheader("👀 Preview of Uploaded Data")
    st.dataframe(df.head(), hide_index=True)

    if not st.session_state.make_predictions:

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
        # print(sp.run("pwd", shell=True, check=True, capture_output=True, text=True))

        # CSV_RAW_INFERENCE_FILENAME = "raw_inference"
        # DEFAULT_CSV_RAW_INFERENCE_PATH = os.path.join(
        #     DATA_FOLDER, CSV_RAW_INFERENCE_FILENAME
        # )

        predictions_file_name = f"predictions_{st.session_state.session_postfix}.csv"
        predictions_file_path = os.path.join("results", predictions_file_name)

        if st.session_state.predictions is None:

            progress_bar = st.progress(0, text="Preparing data file...")
            time.sleep(0.75)

            raw_inference_file_path = os.path.join(
                "data", f"raw_inference_{st.session_state.session_postfix}.csv"
            )
            preprocessed_inference_file_path = os.path.join("data", "inference.csv")
            # TODO: allow user to control output file name?

            df.to_csv(
                raw_inference_file_path,
                index=False,
            )

            progress_bar.progress(1 / 3, text="Preprocessing data...")

            # preprocess inference data
            cmd = f"python preprocess.py --inference-run True --csv-raw-path {raw_inference_file_path}"
            result = sp.run(cmd, shell=True, check=True, capture_output=True, text=True)
            # TODO: control output file name

            progress_bar.progress(2 / 3, text="Making predictions...")

            # Perform prediction
            cmd = f"python predict.py --input-data-path {preprocessed_inference_file_path} --predictions-only True --predictions-file-name {predictions_file_name}"
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

            progress_bar.progress(1.0, text="Done!")
            time.sleep(1)
            progress_bar.empty()

        with open(predictions_file_path, "rb") as file:

            st.subheader("🔮 Preview of CTR Predictions")
            st.session_state.predictions = pd.read_csv(file)

            st.dataframe(
                st.session_state.predictions.head(),
                hide_index=True,
                column_config={0.0: "predictions"},
            )

            # col1, col2 = st.columns(2)
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


# st.set_page_config(layout="wide")

# # Custom CSS to align the title
# st.markdown(
#     """
#     <style>
#     .title {
#         text-align: center;
#     }
#     </style>
#     <style>
#     .sub_title {
#         text-align: center;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )
# st.markdown('<h1 class="title">CTR Predictor 🖱️🔮📊</h1>', unsafe_allow_html=True)
# st.markdown(
#     '<h2 class="sub_title">Upload sessions data and get Click-Through Rate (CTR) predictions!</h2>',
#     unsafe_allow_html=True,
# )
# # st.title("CTR Predictor 🖱️🔮📊")
# # st.markdown("### Upload sessions data and get Click-Through Rate (CTR) predictions!")

# CSS for the app
# # Function to load CSS
# def load_css(file_name):
#     with open(file_name, "r") as f:
#         css = f.read()
#     st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
# # Apply the CSS
# load_css(
#     "./images/style1.css"
# )

#         try:
#             df = pd.read_csv(uploaded_file)
#             st.session_state.df = df
#             st.session_state.uploaded = True
#             st.success("✅ File uploaded successfully!")
#         except Exception as e:
#             st.error(f"❌ Error loading file: {e}")
#             except Exception as e:
#                 st.error(f"❌ An error occurred: {str(e)}")

#                 # try:

#                 # print(result.stdout)  # Output of the command

#                 # except sp.CalledProcessError as e:

#                 #     print(f"Command failed with error code {e.returncode}")
#                 #     print(f"Stdout: {e.stdout}")  # Standard output (if any)
#                 #     print(f"Stderr: {e.stderr}")  # Error output
