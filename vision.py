import os
import cv2
import torch
import numpy as np

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def load_model():
    if not os.path.exists("models/emotion_model.pth"):
        return None
    model = torch.load("models/emotion_model.pth", map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model()

def predict_emotion(image_path):
    if model is None:
        return "Model not available"

    img = cv2.imread(image_path)
    if img is None:
        return "Invalid Image"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48)) / 255.0

    face = np.reshape(face, (1, 1, 48, 48))
    face = torch.tensor(face, dtype=torch.float32)

    with torch.no_grad():
        output = model(face)
        pred = torch.argmax(output).item()

    return emotion_labels[pred]