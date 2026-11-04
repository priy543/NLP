
# ==========================================
# SMS SPAM DETECTION - FASTAPI
# ==========================================

import re
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ==========================================
# 1. LOAD SAVED MODEL
# ==========================================

model = joblib.load("spam_model.pkl")

vectorizer = joblib.load("tfidf_vectorizer.pkl")

print("Model Loaded Successfully!")

# ==========================================
# 2. CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="NLP Spam Detection API",
    description="SMS Spam Detection using TF-IDF and Logistic Regression",
    version="1.0"
)

# ==========================================
# 3. DEFINE INPUT FORMAT
# ==========================================

class SMSRequest(BaseModel):
    message: str

# ==========================================
# 4. TEXT PREPROCESSING
# ==========================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", " ", text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ==========================================
# 5. HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "message": "SMS Spam Detection API is running"
    }

# ==========================================
# 6. PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict(request: SMSRequest):

    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

    message = clean_text(request.message)

    features = vectorizer.transform([message])

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    confidence = float(max(probabilities))

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2)
    }
