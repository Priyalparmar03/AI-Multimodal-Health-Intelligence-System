from transformers import pipeline

# Load model once
classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/bert-base-uncased-emotion"
)

def analyze_text_emotion(text):
    result = classifier(text)
    return result[0]['label'], result[0]['score']