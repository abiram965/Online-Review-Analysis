from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

app = Flask(__name__)

# Load dataset
df = pd.read_csv('flipkart.csv')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/search', methods=['POST'])
def search_product():
    query = request.form.get('query').lower()
    matching_products = df[df['Product_name'].str.lower().str.contains(query)][['ID', 'Product_name']].drop_duplicates()
    
    if matching_products.empty:
        return render_template('home.html', message="No matching products found.")
    else:
        return render_template('results.html', products=matching_products.to_dict(orient='records'))

@app.route('/analyze/<int:product_id>')
def analyze_product(product_id):
    product_reviews = df[df['ID'] == product_id]
    
    if product_reviews.empty:
        return render_template('analysis.html', message="No reviews available for this product.")
    
    # Sentiment analysis
    sentiment_scores = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    analyzer = SentimentIntensityAnalyzer()
    
    for review in product_reviews['Review']:
        sentiment = analyzer.polarity_scores(review)
        if sentiment['compound'] > 0:
            sentiment_scores['Positive'] += 1
        elif sentiment['compound'] == 0:
            sentiment_scores['Neutral'] += 1
        else:
            sentiment_scores['Negative'] += 1
    
    # Save chart
    labels = sentiment_scores.keys()
    sizes = sentiment_scores.values()
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['green', 'gray', 'red'])
    plt.title(f"Sentiment Analysis for Product ID: {product_id}")
    chart_path = f'static/sentiment_chart_{product_id}.png'
    plt.savefig(chart_path)
    plt.close()
    
    return render_template('analysis.html', reviews=product_reviews['Review'].tolist(),
                           sentiment_scores=sentiment_scores, chart_path=chart_path)

if __name__ == '__main__':
    app.run(debug=True)
