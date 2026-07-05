from flask import Flask, render_template, request  # type: ignore
import joblib  # type: ignore
import re
import time
import csv

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # type: ignore[import]
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import]

# -----------------------------
# Load Model & Vectorizer
# -----------------------------
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# -----------------------------
# Load Dataset
# -----------------------------
def load_dataset_texts(*csv_files):
    texts = []
    for csv_file in csv_files:
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts.append(str(row.get("text", "") or ""))
    return texts


dataset_texts = load_dataset_texts("dataset/Fake.csv", "dataset/True.csv")

stop_words = set(ENGLISH_STOP_WORDS)

app = Flask(__name__)

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):
    text = str(text)

    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# -----------------------------
# Clean Dataset Once
# -----------------------------
dataset_texts = [clean_text(text) for text in dataset_texts]

dataset_vectors = vectorizer.transform(dataset_texts)


# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Prediction
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    start_time = time.time()

    news = request.form.get("news", "").strip()

    if news == "":
        return render_template(
            "index.html",
            prediction="⚠ Please enter a news article.",
            confidence=0,
            prediction_time=0,
            color="#f59e0b"
        )

    # Clean input
    cleaned = clean_text(news)

    # Vectorize input
    user_vector = vectorizer.transform([cleaned])

    # -----------------------------
    # Similarity Check
    # -----------------------------
    similarity = cosine_similarity(user_vector, dataset_vectors)

    max_similarity = similarity.max()

    print("Maximum Similarity:", max_similarity)

    THRESHOLD = 0.30

    if max_similarity < THRESHOLD:

        end_time = time.time()

        return render_template(
            "index.html",
            prediction="⚠ DATA NOT FOUND IN DATASET",
            confidence=0,
            prediction_time=round(end_time - start_time, 3),
            color="#f59e0b"
        )

    # -----------------------------
    # ML Prediction
    # -----------------------------
    prediction = model.predict(user_vector)[0]

    probability = model.predict_proba(user_vector).max() * 100

    end_time = time.time()

    prediction_time = round(end_time - start_time, 3)

    if prediction == 1:

        result = "✅ REAL NEWS"

        color = "#22c55e"

    else:

        result = "❌ FAKE NEWS"

        color = "#ef4444"

    return render_template(
        "index.html",
        prediction=result,
        confidence=round(probability, 2),
        prediction_time=prediction_time,
        color=color
    )


# -----------------------------
# Run Application
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
