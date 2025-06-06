import requests
from config import TRADERMADE_API_KEY, TRADERMADE_BASE_URL, X_API_BEARER_TOKEN
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import speech_recognition as sr
import tweepy
import numpy as np

nltk.download('vader_lexicon')

def scrape_news():
    try:
        params = {'api_key': TRADERMADE_API_KEY, 'category': 'forex', 'symbol': 'XAUUSD'}
        response = requests.get(f"{TRADERMADE_BASE_URL}/news", params=params)
        return response.json().get('news', [])
    except Exception:
        return []

def analyze_news_sentiment(news):
    sid = SentimentIntensityAnalyzer()
    if not news:
        return 0.0
    scores = [sid.polarity_scores(article['title'])['compound'] for article in news]
    return np.mean(scores)

def scrape_x_sentiment():
    try:
        auth = tweepy.OAuth2BearerHandler(X_API_BEARER_TOKEN)
        api = tweepy.API(auth)
        tweets = api.search_tweets(q="XAUUSD OR gold", lang="en", count=100)
        sid = SentimentIntensityAnalyzer()
        scores = [sid.polarity_scores(tweet.text)['compound'] for tweet in tweets]
        return np.mean(scores) if scores else 0.0
    except Exception:
        return 0.0

def calculate_sentiment_heatmap(news_sentiment, x_sentiment):
    return (news_sentiment + x_sentiment) / 2

def calculate_correlation_with_lag():
    try:
        # Placeholder: Ambil data historis untuk korelasi (misalnya dari Tradermade)
        response = requests.get(f"{TRADERMADE_BASE_URL}/timeseries", params={
            'currency': 'XAUUSD,DXY,EURUSD,USDJPY',
            'start': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d-%H:%M'),
            'end': datetime.now().strftime('%Y-%m-%d-%H:%M'),
            'period': '1d',
            'api_key': TRADERMADE_API_KEY
        })
        data = response.json()
        prices = {symbol: [float(quote['close']) for quote in data['quotes'] if quote['symbol'] == symbol] for symbol in ['XAUUSD', 'DXY', 'EURUSD', 'USDJPY']}
        correlations = {}
        for symbol1 in prices:
            for symbol2 in prices:
                if symbol1 != symbol2 and len(prices[symbol1]) == len(prices[symbol2]):
                    corr = np.corrcoef(prices[symbol1], prices[symbol2])[0, 1]
                    correlations[f"{symbol1}_{symbol2}"] = corr
        return {
            'EURUSD': correlations.get('XAUUSD_EURUSD', 0.0),
            'USDJPY': correlations.get('XAUUSD_USDJPY', 0.0),
            'DXY': correlations.get('XAUUSD_DXY', 0.0)
        }
    except Exception:
        return {'EURUSD': 0.3, 'USDJPY': 0.2, 'DXY': 0.5}  # Default jika gagal

def get_vix_sentiment():
    try:
        response = requests.get(f"{TRADERMADE_BASE_URL}/vix", params={'api_key': TRADERMADE_API_KEY})
        vix_data = response.json().get('vix', {})
        return float(vix_data.get('sentiment', 0.0)) if vix_data else 0.0
    except Exception:
        return 0.0

def scrape_economic_calendar():
    try:
        response = requests.get(f"{TRADERMADE_BASE_URL}/calendar", params={'api_key': TRADERMADE_API_KEY})
        return response.json().get('events', [])
    except Exception:
        return []

def scrape_cot_report():
    try:
        response = requests.get(f"{TRADERMADE_BASE_URL}/cot", params={'api_key': TRADERMADE_API_KEY, 'symbol': 'XAUUSD'})
        return float(response.json().get('position', 0.0))
    except Exception:
        return 0.0

def speech_to_text_analysis():
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile('fomc_audio.wav') as source:  # Ganti dengan file audio nyata
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            sid = SentimentIntensityAnalyzer()
            return sid.polarity_scores(text)['compound']
    except Exception:
        return 0.0
