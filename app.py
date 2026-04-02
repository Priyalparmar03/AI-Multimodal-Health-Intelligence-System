# AI Health Assistant - Main Application
# This application allows users to input their symptoms in text form and upload a face image to analyze their emotional state. It also predicts possible diseases based on selected symptoms.
import streamlit as st
import pandas as pd
import tempfile

from utils.nlp import analyze_text_emotion
from utils.vision import predict_emotion
from utils.prediction import predict_disease

# Set page configuration
st.set_page_config(page_title="AI Health Assistant", layout="centered")

st.title(" AI Health Assistant")

# TEXT INPUT
text = st.text_area("Enter your symptoms (in text):")

# IMAGE INPUT
image = st.file_uploader("Upload face image", type=["jpg", "png"])

# LOAD SYMPTOM DATA
df = pd.read_csv("data/processed_dataset.csv")
symptom_list = df.columns[:-1]

# SYMPTOM SELECTION
st.subheader(" Select Symptoms")

selected = []
selected_symptoms = []

# Create checkbox UI
for symptom in symptom_list[:20]:  # limit for UI (can increase later)
    if st.checkbox(symptom):
        selected.append(1)
        selected_symptoms.append(symptom)
    else:
        selected.append(0)

# ANALYZE BUTTON
if st.button("Analyze"):

    st.success("Analyzing your health condition...")

    # TEXT EMOTION
    if text.strip() != "":
        st.subheader(" Text Emotion")
        emotion, score = analyze_text_emotion(text)
        st.write(f"Emotion: {emotion} ({score:.2f})")
    else:
        st.warning("Please enter symptom text")

    # FACIAL EMOTION
    if image:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(image.read())
            img_path = tmp.name

        st.subheader(" Facial Emotion")
        face_emotion = predict_emotion(img_path)
        st.write(f"Emotion: {face_emotion}")
    else:
        st.info("No image uploaded")

    # DISEASE PREDICTION
    if sum(selected) == 0:
        st.warning("Please select at least one symptom")
    else:
        st.subheader(" Disease Prediction")

        prediction = predict_disease(selected)
        st.write(f"Predicted Disease: {prediction}")

        st.write("Selected Symptoms:", ", ".join(selected_symptoms))