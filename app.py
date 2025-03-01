from flask import Flask, request, render_template, jsonify
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import os
import spacy
from textblob import TextBlob

app = Flask(__name__)

# Load dataset
df = pd.read_csv("flipkart.csv")

# Load pre-trained models
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# Load NLP model for aspect extraction
nlp = spacy.load("en_core_web_sm")

def extract_aspects(review):
    doc = nlp(review)
    aspects = []
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:  # Extract nouns (product aspects)
            aspects.append(token.text)
    return aspects

def classify_aspect_sentiment(review):
    aspects = extract_aspects(review)
    sentiment_scores = {}
    
    for aspect in aspects:
        sentiment = TextBlob(review).sentiment.polarity  
        sentiment_scores[aspect] = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
    
    return sentiment_scores

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

    for review in reviews:
        sentiment_result = sentiment_model(review)[0]
        sentiment_label = sentiment_result["label"]
        
        if "1" in sentiment_label or "2" in sentiment_label:
            sentiment_scores["Negative"] += 1
        elif "3" in sentiment_label:
            sentiment_scores["Neutral"] += 1
        else:
            sentiment_scores["Positive"] += 1

        emotion_result = emotion_model(review)[0]
        emotion_label = emotion_result["label"]
        emotions[emotion_label] = emotions.get(emotion_label, 0) + 1
        
        aspect_sentiments.update(classify_aspect_sentiment(review))

    labels = sentiment_scores.keys()
    sizes = sentiment_scores.values()
    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=["green", "gray", "red"])
    plt.title("Sentiment Distribution")
    chart_path = "static/sentiment_chart.png"
    plt.savefig(chart_path)
    plt.close()

    return render_template("result.html", reviews=reviews, sentiment_scores=sentiment_scores, emotions=emotions, aspect_sentiments=aspect_sentiments, chart_path=chart_path)

@app.route("/aspect_analysis", methods=["POST"])
def aspect_analysis():
    data = request.get_json()
    review_text = data["review"]

    aspect_sentiments = classify_aspect_sentiment(review_text)

    return jsonify({"review": review_text, "aspect_sentiments": aspect_sentiments})

if __name__ == "__main__":
    app.run(debug=True)
