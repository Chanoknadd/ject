import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Genre to Cluster Predictor", layout="centered")

@st.cache_resource
def load_model():
    with open("random_forest_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Example genre-label mapping (update based on your actual encoding)
genre_map = {
    0: 'Classical', 1: 'Country', 2: 'EDM', 3: 'Folk', 4: 'Gospel',
    5: 'Hip hop', 6: 'Jazz', 7: 'K pop', 8: 'Latin', 9: 'Lofi',
    10: 'Metal', 11: 'Pop', 12: 'R&B', 13: 'Rap', 14: 'Rock', 15: 'Video game music'
}
genre_to_label = {v: k for k, v in genre_map.items()}

# UI
st.title("🎵 Predict Cluster Group from Favorite Genre")

selected_genre = st.selectbox("Select your favorite genre:", list(genre_to_label.keys()))

# Convert to model input
input_label = genre_to_label[selected_genre]
input_df = pd.DataFrame([{"Fav genre_Label": input_label}])

# Predict
prediction = model.predict(input_df)[0]

# Output
st.subheader("🧠 Prediction Result")
st.write(f"Favorite Genre: **{selected_genre}**")
st.write(f"Predicted Cluster Group: **{prediction}**")
