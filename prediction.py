import joblib
import numpy as np
import os

# Load model safely
def load_model():
    if not os.path.exists("models/disease_model.pkl"):
        return None
    return joblib.load("models/disease_model.pkl")

model = load_model()

def predict_disease(symptoms):
    if model is None:
        return "Model not trained"

    symptoms = np.array(symptoms).reshape(1, -1)
    prediction = model.predict(symptoms)
    return prediction[0]