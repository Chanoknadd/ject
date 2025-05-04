import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Group Predictor", layout="centered")

# Load trained model
@st.cache_resource
def load_model():
    with open("random_forest_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Feature columns from the dataset (exclude target labels)
feature_names = [
    'Age', 'Hours per day', 'While working', 'Instrumentalist', 'Composer',
    'Foreign languages', 'Frequency [Classical]', 'Frequency [Country]',
    'Frequency [EDM]', 'Frequency [Folk]', 'Frequency [Gospel]',
    'Frequency [Hip hop]', 'Frequency [Jazz]', 'Frequency [K pop]',
    'Frequency [Latin]', 'Frequency [Lofi]', 'Frequency [Metal]',
    'Frequency [Pop]', 'Frequency [R&B]', 'Frequency [Rap]',
    'Frequency [Rock]', 'Frequency [Video game music]'
]

# Sidebar input
st.sidebar.header("🎧 Input Features")
input_data = {}
for feature in feature_names:
    input_data[feature] = st.sidebar.number_input(
        label=feature,
        min_value=0,
        max_value=10,
        value=5
    )

# Prediction
input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)[0]

# Output
st.title("🎯 Cluster Group Prediction")
st.write("Based on your input features:")
st.write(f"**Predicted Cluster Group:** `{prediction}`")
