import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

st.set_page_config(page_title="Group Predictor", layout="centered")

# Load data
@st.cache_data
def load_data():
    file_path = "dataset2_clustered (1).csv"
    df = pd.read_csv(file_path)
    return df

st.title("🎯 Predict Group / Genre from Features")

df = load_data()

# Detect target column
target_col = None
for col in df.columns:
    if "cluster" in col.lower() or "genre" in col.lower() or "group" in col.lower():
        target_col = col
        break

if not target_col:
    st.error("No target column (cluster/genre/group) found in dataset.")
    st.stop()

st.success(f"Target column detected: **{target_col}**")

# Prepare data
X = df.drop(columns=[target_col])
y = df[target_col]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Sidebar input
st.sidebar.header("🔢 Enter Feature Values")
input_data = {}
for col in X.columns:
    input_data[col] = st.sidebar.number_input(
        label=f"{col}", 
        min_value=float(df[col].min()), 
        max_value=float(df[col].max()), 
        value=float(df[col].mean())
    )

# Predict
input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)[0]

# Output
st.subheader("🧠 Prediction")
st.write(f"Predicted Group/Genre: **{prediction}**")
