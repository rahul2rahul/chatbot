import json
import pickle
import random
import re
import os
import numpy as np
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import joblib

from flask import Flask, request, jsonify
from flask_cors import CORS

# Download NLTK data on startup
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

app = Flask(__name__)
CORS(app)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Load models
BASE = os.path.dirname(os.path.abspath(__file__))
chatbot_model  = joblib.load(os.path.join(BASE, "model", "chatbot_model.pkl"))
career_model   = joblib.load(os.path.join(BASE, "model", "career_model.pkl"))
label_encoder  = joblib.load(os.path.join(BASE, "model", "label_encoder.pkl"))
course_reasons = pd.read_csv(os.path.join(BASE, "model", "course_reasons.csv"))

with open(os.path.join(BASE, "model", "intents_dict.pkl"), "rb") as f:
    intents_dict = pickle.load(f)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
               if t not in stop_words and len(t) > 1]
    return " ".join(tokens)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Ardent Chatbot API is running", "version": "1.0"})

@app.route("/chat", methods=["POST"])
def chat():
    """Chatbot endpoint - handles course info & FAQ queries"""
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "message field required"}), 400

    user_msg = str(data["message"]).strip()
    if not user_msg:
        return jsonify({"error": "empty message"}), 400

    processed = preprocess_text(user_msg)
    proba = chatbot_model.predict_proba([processed])[0]
    max_proba = float(max(proba))
    predicted_tag = chatbot_model.classes_[np.argmax(proba)]

    threshold = 0.25
    if max_proba < threshold:
        predicted_tag = "not_understood"

    response = random.choice(intents_dict[predicted_tag]["responses"])

    return jsonify({
        "response": response,
        "tag": predicted_tag,
        "confidence": round(max_proba, 3)
    })

@app.route("/recommend", methods=["POST"])
def recommend():
    """Career guidance endpoint - recommends courses based on profile"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    # Build profile text from user inputs
    parts = []
    fields = ["background","field","current_role","experience_years",
               "skills","interest","goal","education"]
    for key in fields:
        value = data.get(key, "")
        if value and str(value).lower() not in ["none","nan",""]:
            parts.append(f"{key} {value}")

    if not parts:
        return jsonify({"error": "At least one profile field required"}), 400

    profile_text = preprocess_text(" ".join(parts))

    # Top 3 recommendations
    proba = career_model.predict_proba([profile_text])[0]
    top_indices = np.argsort(proba)[::-1][:3]

    recommendations = []
    for idx in top_indices:
        course = label_encoder.classes_[idx]
        confidence = float(proba[idx])

        matching = course_reasons[course_reasons["primary_course"] == course]
        reason = matching["reason"].iloc[0] if len(matching) > 0 else "Highly relevant for your profile"
        all_courses = matching["recommended_courses"].iloc[0] if len(matching) > 0 else course

        recommendations.append({
            "primary_course": course,
            "related_courses": all_courses,
            "reason": reason,
            "confidence": round(confidence, 3)
        })

    return jsonify({
        "recommendations": recommendations,
        "profile_summary": " ".join(parts)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)