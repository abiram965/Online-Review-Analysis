from flask import Flask, request, render_template, jsonify
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import os
import spacy
from textblob import TextBlob
from langdetect import detect
from deep_translator import GoogleTranslator
import requests
from flask_socketio import SocketIO, emit


app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading")

# Load dataset
df = pd.read_csv("flipkart.csv")

# Load pre-trained models
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Set Google Gemini API Key
GEMINI_API_KEY = ""

# Function to detect language and translate
def detect_and_translate(text):
    try:
        lang = detect(text)
        if lang != "en":
            translated_text = GoogleTranslator(source='auto', target='en').translate(text)
            return translated_text, lang
        return text, "en"
    except:
        return text, "unknown"

# Extract aspects for Aspect-Based Sentiment Analysis
def extract_aspects(review):
    doc = nlp(review)
    aspects = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    return aspects

def classify_aspect_sentiment(review):
    aspects = extract_aspects(review)
    sentiment_scores = {aspect: "Neutral" for aspect in aspects}
    
    for aspect in aspects:
        sentiment = TextBlob(review).sentiment.polarity  
        sentiment_scores[aspect] = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
    
    return sentiment_scores

# Chatbot function using Google Gemini API
def chatbot_response(user_input):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": user_input}]}]}

    try:
        response = requests.post(url, json=data, headers=headers)
        response_json = response.json()
        return response_json["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return "Sorry, I am currently unavailable."

# Handle chat messages
@socketio.on("message")
def handle_message(data):
    user_input = data["message"]
    bot_response = chatbot_response(user_input)
    emit("response", {"message": bot_response})

@app.route("/")
def home():
    products = df[['ID', 'Product_name']].drop_duplicates()
    return render_template("home.html", products=products.to_dict(orient='records'))

@app.route("/analyze", methods=["POST"])
def analyze():
    product_id = request.form.get("product_id")
    reviews = df[df["ID"] == int(product_id)]["Review"].tolist()

    if not reviews:
        return render_template("result.html", message="No reviews found.")

    sentiment_scores = {"Positive": 0, "Neutral": 0, "Negative": 0}
    emotions = {}
    aspect_sentiments = {}
    translated_reviews = []

    for review in reviews:
        translated_review, original_language = detect_and_translate(review)
        translated_reviews.append({"original": review, "translated": translated_review, "language": original_language})

        sentiment_result = sentiment_model(translated_review)[0]
        sentiment_label = sentiment_result["label"]
        
        if "1" in sentiment_label or "2" in sentiment_label:
            sentiment_scores["Negative"] += 1
        elif "3" in sentiment_label:
            sentiment_scores["Neutral"] += 1
        else:
            sentiment_scores["Positive"] += 1

        emotion_result = emotion_model(translated_review)[0]
        emotions[emotion_result["label"]] = emotions.get(emotion_result["label"], 0) + 1
        
        aspect_sentiments.update(classify_aspect_sentiment(translated_review))

    # Generate Sentiment Pie Chart
    labels = sentiment_scores.keys()
    sizes = sentiment_scores.values()
    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=["green", "gray", "red"])
    plt.title("Sentiment Distribution")
    chart_path = "static/sentiment_chart.png"
    plt.savefig(chart_path)
    plt.close()

    return render_template("result.html", reviews=translated_reviews, sentiment_scores=sentiment_scores, emotions=emotions, aspect_sentiments=aspect_sentiments, chart_path=chart_path)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))


