from flask import Flask, request, render_template, jsonify, redirect, url_for
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
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app, async_mode="eventlet")

##checking
# Load data and models
df = pd.read_csv("flipkart.csv")
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
nlp = spacy.load("en_core_web_sm")

# Google Gemini config
dotenv.load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

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

def get_review_aspects(reviews):
    """Extract aspects from reviews and format them for display"""
    formatted_reviews = []
    for review in reviews:
        translated_review, original_language = detect_and_translate(review)
        
        # Get sentiment for the review
        sentiment_result = sentiment_model(translated_review)[0]
        label = sentiment_result["label"]
        if "1" in label or "2" in label:
            sentiment = "Negative"
        elif "3" in label:
            sentiment = "Neutral"
        else:
            sentiment = "Positive"
        
        # Extract aspects and their sentiments
        aspects = extract_aspects(translated_review)
        aspect_sentiments = []
        for aspect in aspects[:5]:  # Limit to 5 aspects per review for UI
            sentiment_text = TextBlob(translated_review).sentiment.polarity
            aspect_sentiment = "Positive" if sentiment_text > 0 else "Negative" if sentiment_text < 0 else "Neutral"
            aspect_sentiments.append({"name": aspect, "sentiment": aspect_sentiment})
        
        formatted_reviews.append({
            "original": review,
            "translated": translated_review,
            "language": original_language,
            "sentiment": sentiment,
            "aspects": aspect_sentiments
        })
    
    return formatted_reviews

@socketio.on("message")
def handle_message(data):
    user_input = data["message"]
    product_id = data.get("product_id", 0)
    response = chatbot_response(user_input, product_id)
    emit("response", {"message": response})

@app.route("/")
def home():
    products = df[['ID', 'Product_name']].drop_duplicates()
    
    # Add some stats for the dashboard
    total_reviews = len(df)
    languages = set()
    positive_count = 0
    
    # Sample 100 reviews for stats calculation to avoid performance issues
    sample_reviews = df["Review"].sample(min(100, len(df))).tolist()
    
    for review in sample_reviews:
        _, lang = detect_and_translate(review)
        languages.add(lang)
        
        # Quick sentiment check
        sentiment_result = sentiment_model(review)[0]
        label = sentiment_result["label"]
        if "4" in label or "5" in label:
            positive_count += 1
    
    positive_percentage = int((positive_count / len(sample_reviews)) * 100)
    
    return render_template(
        "home.html", 
        products=products.to_dict(orient='records'),
        total_reviews=total_reviews,
        total_languages=len(languages),
        positive_percentage=positive_percentage
    )

@app.route("/analyze", methods=["POST"])
def analyze():
    product_id = request.form.get("product_id")
    
    if not product_id:
        return render_template("result.html", message="No product selected.")
    
    # Option to show the analyzing page first
    if request.form.get("show_analysis_progress", "false") == "true":
        return render_template("analyze.html", product_id=product_id)
    
    # Otherwise process immediately
    return process_analysis(product_id)

@app.route("/process", methods=["GET"])
def process():
    product_id = request.args.get("product_id")
    if not product_id:
        return redirect(url_for("home"))
    
    return process_analysis(product_id)

def process_analysis(product_id):
    reviews = df[df["ID"] == int(product_id)]["Review"].tolist()

    if not reviews:
        return render_template("result.html", message="No reviews found.")

    # Process sentiment distribution
    sentiment_scores = {"Positive": 0, "Neutral": 0, "Negative": 0}
    emotions = {}
    aspect_sentiments = {}

    # Get formatted reviews with aspects
    formatted_reviews = get_review_aspects(reviews[:20])  # Limit to 20 reviews for UI
    
    # Count sentiments from formatted reviews
    for review in formatted_reviews:
        sentiment_scores[review["sentiment"]] += 1
        
        # Process emotions
        emotion_result = emotion_model(review["translated"])[0]
        emotions[emotion_result["label"]] = emotions.get(emotion_result["label"], 0) + 1
        
        # Collect aspects
        for aspect_data in review["aspects"]:
            aspect = aspect_data["name"]
            sentiment = aspect_data["sentiment"]
            aspect_sentiments[aspect] = sentiment

    # Create sentiment chart
    labels = sentiment_scores.keys()
    sizes = sentiment_scores.values()
    plt.figure(figsize=(6,6))
    colors = ["#4cc9a4", "#f9c74f", "#f94144"]  # green, yellow, red
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title("Sentiment Distribution")
    chart_path = "static/sentiment_chart.png"
    plt.savefig(chart_path)
    plt.close()

    # Define emotion colors for the template
    emotion_colors = [
        'rgba(255, 99, 132, 0.7)',   # red
        'rgba(54, 162, 235, 0.7)',   # blue
        'rgba(255, 206, 86, 0.7)',   # yellow
        'rgba(75, 192, 192, 0.7)',   # green
        'rgba(153, 102, 255, 0.7)',  # purple
        'rgba(255, 159, 64, 0.7)'    # orange
    ]

    return render_template(
        "result.html", 
        reviews=formatted_reviews, 
        sentiment_scores=sentiment_scores,
        emotions=emotions, 
        aspect_sentiments=aspect_sentiments,
        chart_path=chart_path, 
        product_id=product_id,
        emotion_colors=emotion_colors
    )

@app.route("/result", methods=["GET"])
def result():
    product_id = request.args.get("id")
    if not product_id:
        return redirect(url_for("home"))
    
    return process_analysis(product_id)

if __name__ == "__main__":
    socketio.run(app, debug=True)