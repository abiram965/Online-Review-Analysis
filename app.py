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
import google.generativeai as genai
import dotenv

app = Flask(__name__)
socketio = SocketIO(app, async_mode="gevent")

# Load data and models
df = pd.read_csv("flipkart.csv")
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
nlp = spacy.load("en_core_web_sm")

# Google Gemini config
dotenv.load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")


def detect_and_translate(text):
    try:
        lang = detect(text)
        if lang != "en":
            translated_text = GoogleTranslator(source='auto', target='en').translate(text)
            return translated_text, lang
        return text, "en"
    except:
        return text, "unknown"

def extract_aspects(review):
    doc = nlp(review)
    return [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]

def classify_aspect_sentiment(review):
    aspects = extract_aspects(review)
    sentiment_scores = {aspect: "Neutral" for aspect in aspects}
    for aspect in aspects:
        sentiment = TextBlob(review).sentiment.polarity
        sentiment_scores[aspect] = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
    return sentiment_scores

def chatbot_response(user_input, product_id):
    product_reviews = df[df["ID"] == int(product_id)]["Review"].tolist()
    context_reviews = "\n".join(product_reviews[:10])  # Use top 10 reviews for context

    prompt = f"""
You are an AI assistant built to answer questions specifically about the 'E-Commerce Reviews Sentiment Analysis' project.
The user is analyzing Product ID {product_id}.
Here are some real reviews from that product:
{context_reviews}

Only respond to questions about this product or the sentiment/emotion analysis process.
If asked about anything else, say you're only trained to discuss this project.

User: {user_input}
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return "Sorry, I am currently unavailable."

@socketio.on("message")
def handle_message(data):
    user_input = data["message"]
    product_id = data.get("product_id", 0)
    response = chatbot_response(user_input, product_id)
    emit("response", {"message": response})

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
        label = sentiment_result["label"]
        if "1" in label or "2" in label:
            sentiment_scores["Negative"] += 1
        elif "3" in label:
            sentiment_scores["Neutral"] += 1
        else:
            sentiment_scores["Positive"] += 1

        emotion_result = emotion_model(translated_review)[0]
        emotions[emotion_result["label"]] = emotions.get(emotion_result["label"], 0) + 1

        aspect_sentiments.update(classify_aspect_sentiment(translated_review))

    labels = sentiment_scores.keys()
    sizes = sentiment_scores.values()
    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=["green", "gray", "red"])
    plt.title("Sentiment Distribution")
    chart_path = "static/sentiment_chart.png"
    plt.savefig(chart_path)
    plt.close()

    return render_template("result.html", reviews=translated_reviews, sentiment_scores=sentiment_scores,
                           emotions=emotions, aspect_sentiments=aspect_sentiments,
                           chart_path=chart_path, product_id=product_id)

if __name__ == "__main__":
    socketio.run(app, debug=True)
