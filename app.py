import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Predict Group", layout="centered")

# Load pre-trained model
@st.cache_resource
def load_model():
    with open("random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Define expected features and ranges (example values — change these to match your model!)
feature_info = {
    "tempo": (60, 200, 120),
    "energy": (0.0, 1.0, 0.5),
    "danceability": (0.0, 1.0, 0.5),
    "valence": (0.0, 1.0, 0.5),
    "loudness": (-60, 0, -20)
}

# UI
st.title("🎵 Predict Group / Genre from Features")
st.sidebar.header("🔢 Enter Song Features")

input_data = {}
for feature, (min_val, max_val, default_val) in feature_info.items():
    input_data[feature] = st.sidebar.number_input(
        label=feature.capitalize(),
        min_value=min_val,
        max_value=max_val,
        value=default_val
    )

# Prediction
input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)[0]

# Output
st.subheader("🧠 Prediction Result")
st.write(f"Predicted Group/Genre: **{prediction}**")
