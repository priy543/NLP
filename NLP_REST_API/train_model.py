
# ==========================================
# SMS SPAM DETECTION - MODEL TRAINING
# ==========================================

import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv(
    "spam.csv",
    sep="\t",
    header=None,
    encoding="latin-1"
)

df.columns = ["label", "message"]

print("========== DATASET INFORMATION ==========")
print("Shape:", df.shape)
print(df.head())

# ==========================================
# 2. DATA CLEANING
# ==========================================

df = df.dropna()

df["label"] = df["label"].astype(str).str.lower().str.strip()
df["message"] = df["message"].astype(str)

df = df[df["label"].isin(["ham", "spam"])]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["message"] = df["message"].apply(clean_text)
df = df[df["message"] != ""]

print("\n========== CLEANED DATA ==========")
print("Shape:", df.shape)
print(df.head())

print("\n========== LABEL DISTRIBUTION ==========")
print(df["label"].value_counts())

# ==========================================
# 3. TRAIN TEST SPLIT
# ==========================================

X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ==========================================
# 4. TF-IDF FEATURE EXTRACTION
# ==========================================

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF Feature Shape:", X_train_tfidf.shape)

# ==========================================
# 5. TRAIN LOGISTIC REGRESSION
# ==========================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

print("\nModel Training Completed!")

# ==========================================
# 6. MODEL EVALUATION
# ==========================================

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))

# ==========================================
# 7. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["ham", "spam"]
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

disp.plot(cmap="Blues", values_format="d")
plt.title("SMS Spam Detection Confusion Matrix")
plt.show()

# ==========================================
# 8. SAVE MODEL AND VECTORIZER
# ==========================================

joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\n========== MODEL SAVED ==========")
print("spam_model.pkl")
print("tfidf_vectorizer.pkl")

# ==========================================
# 9. TEST SAMPLE MESSAGE
# ==========================================

sample = "Congratulations! You won a cash prize. Click now!"

sample_cleaned = clean_text(sample)
sample_tfidf = vectorizer.transform([sample_cleaned])

prediction = model.predict(sample_tfidf)[0]

probabilities = model.predict_proba(sample_tfidf)[0]
confidence = max(probabilities)

print("\n========== SAMPLE PREDICTION ==========")
print("Message:", sample)
print("Prediction:", prediction)
print("Confidence:", round(confidence, 2))
import joblib

joblib.dump(model, "spam_model.pkl")

print("spam_model.pkl saved successfully!")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("tfidf_vectorizer.pkl saved successfully!")