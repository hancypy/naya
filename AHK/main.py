from datetime import datetime
from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import AsyncImage
from kivy.uix.label import Label
from kivymd.toast import toast
from kivymd.uix.button import MDRaisedButton, MDFloatingActionButton, MDIconButton, MDRoundFlatIconButton
from kivymd.uix.expansionpanel import MDExpansionPanel, MDExpansionPanelThreeLine
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.list import OneLineListItem
from kivymd.uix.screen import MDScreen
from kivymd.uix.textfield import MDTextField
from kivy.utils import get_color_from_hex
from kivymd.uix.toolbar import MDTopAppBar
from static_content import courses
from kivy.uix.scrollview import ScrollView
from kivymd.uix.screenmanager import MDScreenManager
from posstock import CompanyListScreen, ProductListScreen, TransactionScreen
import pyrebase
from threading import Thread
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2 import id_token
from google.auth.transport.requests import Request
from kivy.utils import platform
from expiry import is_token_expired
import json
import os
from kivymd.uix.menu import MDDropdownMenu
from kivy.uix.screenmanager import Screen
from functools import partial
import pandas_ta as ta
import re
import threading
import numpy as np
from kivy.metrics import dp
from kivy.properties import StringProperty
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFlatButton
from kivymd.uix.card import MDCard, MDSeparator
from kivymd.uix.dialog import MDDialog
from kivymd.uix.label import MDLabel
from kivymd.uix.scrollview import MDScrollView
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands
from concurrent.futures import  as_completed
import logging
from collections import defaultdict
from ta.momentum import RSIIndicator, StochasticOscillator
import math
from afinn import Afinn
import sys
from kivy.core.window import Window
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import praw
import requests
from kivymd.uix.progressbar import MDProgressBar
from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator
from vaderSentiment import SentimentIntensityAnalyzer
from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.clock import mainthread, Clock
if platform == "android":
    from jnius import autoclass, PythonJavaClass, java_method
    from android.storage import primary_external_storage_path
    from android.permissions import check_permission, request_permissions, Permission
if platform == 'android':
    from android.permissions import request_permissions, Permission

    request_permissions([Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE])


#Window.size = (394, 600)


# CoinMarketCap API Configuration
CMC_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
CMC_HEADERS = {
    "X-CMC_PRO_API_KEY": CMC_API_KEY,
    "Accepts": "application/json"
}
CMC_PARAMS = {
    "start": "1",  # Start from the top-ranked coins
    "limit": "1000",  # Adjust as needed based on your API tier
    "convert": "USD"
}
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Total number of top coins to consider for technical analysis
SUBMISSIONS_LIMIT = 100  # Number of Reddit submissions to analyze per coin
TOP_N_COINS = 20
FINAL_TOP_N = 10
TECHNICAL_ANALYSIS_CANDIDATES = 10
SENTIMENT_WEIGHT = 0.49
PERFORMANCE_WEIGHT = 0.51
MAX_WORKERS = 10  # Adjust based on your system's capability


# ===========================================================
# -----------------------
# SETTINGS & PARAMETERS
# -----------------------
#COIN_NAME = 'not'  # Change to desired coin name (e.g., 'ethereum')
VS_CURRENCY = 'usd'
DAYS = 30  # Number of days to fetch (CoinGecko supports specific values like 30, 90, etc.)
RSI_PERIOD = 14
EMA_PERIOD = 20
ADX_PERIOD = 14  # For ADX indicator

# RSI thresholds
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# SMC parameters: look back this many data points for swing highs/lows
SWING_LOOKBACK = 20
PRICE_TOLERANCE = 0.005  # within 0.5% of a key level is considered "near" that level

# Fibonacci tolerance factor (can be adjusted separately if needed)
FIB_TOLERANCE = PRICE_TOLERANCE
# Reddit API Configuration
executor = ThreadPoolExecutor(max_workers=5)

def search_coin_id(coin_name):
    """
    Search for the coin ID on CoinGecko based on a query.
    """
    url = f'https://api.coingecko.com/api/v3/search?query={coin_name}'
    params = {"query": coin_name.lower()}
    if COINGECKO_API_KEY:
        params["x_cg_demo_api_key"] = COINGECKO_API_KEY
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        return None
    data = resp.json()
    coins = data.get("coins", [])
    if not coins:
        return None
    coins.sort(key=lambda c: c.get("score", 0), reverse=True)
    # attempt exact matches
    for c in coins:
        if c["id"].lower() == coin_name.lower():
            return c["id"]
    for c in coins:
        if c["name"].lower() == coin_name.lower():
            return c["id"]
    for c in coins:
        if c["symbol"].lower() == coin_name.lower():
            return c["id"]
    return coins[0]["id"]

def fetch_ohlc_data(coin_id, vs_currency='usd', days=30):
    """
    Fetch OHLC data from CoinGecko for the specified coin_id.
    Note: The CoinGecko /ohlc endpoint supports days in {1, 7, 14, 30, 90, 180, 365, max}.
    Data format returned: [timestamp, open, high, low, close]
    """
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc'
    params = {'vs_currency': vs_currency, 'days': days,"x_cg_demo_api_key":COINGECKO_API_KEY}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close'])
        # Convert timestamp (in ms) to datetime
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        return df.drop(columns=['Timestamp'])
    else:
        print("Error fetching data:", response.status_code, response.text)
        return None

def fetch_market_chart_data(coin_id, vs_currency='usd', days=30):
    """
    Fetch historical market chart data, including prices and volume, from CoinGecko.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency': vs_currency, 'days': days}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()

        # Check if volume data is available
        if 'total_volumes' not in data or len(data['total_volumes']) == 0:
            print(f"Warning: No volume data available for {coin_id}")
            return None  # Avoid processing NaN data

        prices = pd.DataFrame(data['prices'], columns=['Timestamp', 'Price'])
        volumes = pd.DataFrame(data['total_volumes'], columns=['Timestamp', 'Volume'])

        # Convert timestamps
        prices['Date'] = pd.to_datetime(prices['Timestamp'], unit='ms')
        volumes['Date'] = pd.to_datetime(volumes['Timestamp'], unit='ms')

        # Merge price and volume into one DataFrame
        df = prices.merge(volumes, on='Date', how='left')
        df.set_index('Date', inplace=True)
        df.drop(columns=['Timestamp_x', 'Timestamp_y'], inplace=True, errors='ignore')

        return df
    else:
        print("Error fetching market chart data:", response.status_code, response.text)
        return None

def calculate_fibonacci_levels(recent_high, recent_low):
    """
    Calculate common Fibonacci retracement levels between the recent swing high and swing low.
    Returns a dictionary of levels.
    """
    diff = recent_high - recent_low
    levels = {
        '0.0%': recent_high,
        '23.6%': recent_high - 0.236 * diff,
        '38.2%': recent_high - 0.382 * diff,
        '50.0%': recent_high - 0.5 * diff,
        '61.8%': recent_high - 0.618 * diff,
        '78.6%': recent_high - 0.786 * diff,
        '100.0%': recent_low
    }
    return levels

def check_price_near_fib(current_price, fib_levels, tolerance=FIB_TOLERANCE):
    """
    Check if the current price is near any Fibonacci retracement level within a given tolerance.
    Returns a tuple (signal, level, value) if near, otherwise (None, None, None).
    """
    for level, value in fib_levels.items():
        if abs(current_price - value) / current_price <= tolerance:
            return f"Price is near Fibonacci level {level}", level, value
    return None, None, None

def get_smc_signal(df, lookback=SWING_LOOKBACK, tolerance=PRICE_TOLERANCE):
    """
    Identify a basic SMC-like signal by checking if the current price is near a recent swing high or low.
    Returns the SMC signal, recent high, and recent low.
    """
    current_price = df['Close'].iloc[-1]
    recent_data = df[-lookback:]

    recent_high = recent_data['High'].max()
    recent_low = recent_data['Low'].min()

    near_high = abs(current_price - recent_high) / current_price <= tolerance
    near_low = abs(current_price - recent_low) / current_price <= tolerance

    if near_low and not near_high:
        signal = 'Bullish SMC (Support)'
    elif near_high and not near_low:
        signal = 'Bearish SMC (Resistance)'
    elif near_high and near_low:
        signal = 'Indecision (Both Sides)'
    else:
        signal = 'Neutral SMC'

    return signal, recent_high, recent_low

def human_readable(number):
    if number >= 1_000_000_000_000:  # Trillion
        return f"{number / 1_000_000_000_000:.2f}T"
    elif number >= 1_000_000_000:  # Billion
        return f"{number / 1_000_000_000:.2f}B"
    elif number >= 1_000_000:  # Million
        return f"{number / 1_000_000:.2f}M"
    else:
        if number >= 0.01:
            return f"{number:.4f}"  # For numbers >= 0.01, keep exactly four decimal places
        else:
            return "{:.7f}".format(number).rstrip('0')  # Trim trailing zeros for small numbers

def main2(COIN_NAME):

    # -----------------------
    # FETCH DATA FROM COINGECKO
    # -----------------------
    coin_id = search_coin_id(COIN_NAME)
    if not coin_id:
        raise ValueError(f"Coin '{COIN_NAME}' not found on CoinGecko.")

    data = fetch_ohlc_data(coin_id, vs_currency=VS_CURRENCY, days=DAYS)
    if data is None or data.empty:
        raise ValueError("No data was fetched from CoinGecko.")
    market_data = fetch_market_chart_data(coin_id, VS_CURRENCY, DAYS)

    if market_data is None:
        raise ValueError("Failed to retrieve market chart data.")
    # Convert both data and market_data to daily timestamps
    data.index = pd.to_datetime(data.index).floor('D')
    market_data.index = pd.to_datetime(market_data.index).floor('D')

    # Aggregate volume to daily sum
    market_data_daily = market_data.resample('D').sum()

    # Merge volume into data
    data = data.merge(market_data_daily[['Volume']], left_index=True, right_index=True, how='left')

    # Ensure volume is numeric and prevent NaN issues
    data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce').fillna(0)

    # Calculate Volume SMA with min_periods=1
    VOLUME_SMA_PERIOD = 20
    data['Volume_SMA'] = data['Volume'].rolling(VOLUME_SMA_PERIOD, min_periods=1).mean()



    # Volume signal: If the latest volume is above SMA, it indicates strong participation
    latest_volume = data['Volume'].iloc[-1]
    latest_volume_sma = data['Volume_SMA'].iloc[-1]


    # -----------------------
    # CALCULATE BASE INDICATORS (EMA, RSI)
    # -----------------------
    data['EMA'] = EMAIndicator(close=data['Close'], window=EMA_PERIOD).ema_indicator()
    data['RSI'] = RSIIndicator(close=data['Close'], window=RSI_PERIOD).rsi()

    # -----------------------
    # CALCULATE ADDITIONAL FILTERS
    # -----------------------
    # ADX
    data['ADX'] = ADXIndicator(high=data['High'], low=data['Low'], close=data['Close'], window=ADX_PERIOD).adx()

    # MACD
    macd = MACD(close=data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_signal'] = macd.macd_signal()
    data['MACD_diff'] = macd.macd_diff()

    # -----------------------
    # GET THE LATEST DATA POINT
    # -----------------------
    latest = data.iloc[-1]
    price = latest['Close']
    ema = latest['EMA']
    rsi = latest['RSI']
    adx = latest['ADX']
    macd_value = latest['MACD']
    macd_signal_val = latest['MACD_signal']

    # For candlestick pattern, check if the latest candle is bullish (close > open)
    candle_signal = 'Bullish' if latest['Close'] > latest['Open'] else 'Bearish'

    # -----------------------
    # DERIVE INDIVIDUAL SIGNALS
    # -----------------------
    # EMA signal: Price above EMA is bullish; below is bearish.
    ema_signal = 'Bullish' if price > ema else 'Bearish'

    # RSI signal: Overbought means bearish; oversold means bullish; otherwise neutral.
    if rsi > RSI_OVERBOUGHT:
        rsi_signal = 'Bearish (Overbought)'
    elif rsi < RSI_OVERSOLD:
        rsi_signal = 'Bullish (Oversold)'
    else:
        rsi_signal = 'Neutral'

    # SMC signal:
    smc_signal, recent_high, recent_low = get_smc_signal(data)

    # Fibonacci retracement based on recent swing high/low (from SMC)
    fib_levels = calculate_fibonacci_levels(recent_high, recent_low)
    fib_signal, fib_level_key, fib_value = check_price_near_fib(price, fib_levels)

    # MACD filter: Bullish if MACD line is above its signal line.
    macd_filter = 'Bullish' if macd_value > macd_signal_val else 'Bearish'

    # ADX filter is not directional by itself—it tells us if the trend is strong.
    adx_filter = 'Strong Trend' if adx > 25 else 'Weak Trend'

    # -----------------------
    # COMBINE SIGNALS USING A VOTING SYSTEM
    # -----------------------
    # Each filter contributes one "vote" for bullish or bearish.
    bullish_votes = 0
    bearish_votes = 0
    reasons = []

    # EMA vote
    if ema_signal == 'Bullish':
        bullish_votes += 1
        reasons.append("Price is above EMA.")
    else:
        bearish_votes += 1
        reasons.append("Price is below EMA.")

    # RSI vote (only count if clearly overbought/oversold)
    if 'Bullish' in rsi_signal:
        bullish_votes += 1
        reasons.append("RSI indicates oversold conditions.")
    elif 'Bearish' in rsi_signal:
        bearish_votes += 1
        reasons.append("RSI indicates overbought conditions.")
    if latest_volume > latest_volume_sma:
        volume_signal = 'Bullish (High Volume)'
        bullish_votes += 1
        reasons.append("Volume is above average, indicating strong buying interest.")
    else:
        volume_signal = 'Bearish (Low Volume)'
        bearish_votes += 1
        reasons.append("Volume is below average, indicating weak market participation.")
    # SMC vote
    if 'Bullish' in smc_signal:
        bullish_votes += 1
        reasons.append("SMC indicates support.")
    elif 'Bearish' in smc_signal:
        bearish_votes += 1
        reasons.append("SMC indicates resistance.")

    # Fibonacci vote (if near a key level)
    if fib_signal:
        # Typically, being near the 100.0% level (recent low) is seen as support
        if fib_level_key == '100.0%':
            bullish_votes += 1
            reasons.append(f"Price is near Fibonacci support level {fib_level_key}.")
        # If near the top, it may act as resistance
        elif fib_level_key == '0.0%':
            bearish_votes += 1
            reasons.append(f"Price is near Fibonacci resistance level {fib_level_key}.")
        else:
            # For intermediate levels, you can decide to add a vote if you believe it’s significant.
            reasons.append(f"Price is near intermediate Fibonacci level {fib_level_key}.")

    # MACD vote
    if macd_filter == 'Bullish':
        bullish_votes += 1
        reasons.append("MACD indicates bullish momentum.")
    else:
        bearish_votes += 1
        reasons.append("MACD indicates bearish momentum.")

    # Candlestick vote (latest candle)
    if candle_signal == 'Bullish':
        bullish_votes += 1
        reasons.append("Latest candlestick is bullish.")
    else:
        bearish_votes += 1
        reasons.append("Latest candlestick is bearish.")

    # -----------------------
    # FINAL COMBINED SIGNAL
    # -----------------------
    if bullish_votes > bearish_votes:
        final_trend = 'Bullish'
    elif bearish_votes > bullish_votes:
        final_trend = 'Bearish'
    else:
        final_trend = 'Sideways'

    # Create a reason summary from the filters
    reason_summary = " ".join(reasons)





    # -----------------------
    # PRINT RESULTS
    # -----------------------





    # Example Usage in the print function:
    print(f"Coin: {COIN_NAME.title()} ({coin_id})")
    print(f"Latest Price: {human_readable(price)} {VS_CURRENCY.upper()}")
    print(f"EMA ({EMA_PERIOD} period): {human_readable(ema)} -> Signal: {ema_signal}")
    print(f"RSI ({RSI_PERIOD} period): {human_readable(rsi)} -> Signal: {rsi_signal}")
    print(f"ADX ({ADX_PERIOD} period): {human_readable(adx)} -> Trend Strength: {adx_filter}")
    print(f"MACD: {human_readable(macd_value)} vs Signal: {human_readable(macd_signal_val)} -> Filter: {macd_filter}")
    print(f"Candlestick Signal: {candle_signal}")
    print(
        f"SMC Signal: {smc_signal} (Recent High: {human_readable(recent_high)}, Recent Low: {human_readable(recent_low)})")
    print("Fibonacci Levels:")
    for level, value in fib_levels.items():
        print(f"  {level}: {human_readable(value)}")
    if fib_signal:
        print(f"Fibonacci Signal: {fib_signal} (Level {fib_level_key}: {human_readable(fib_value)})")
    else:
        print("Fibonacci Signal: No key Fib level near current price")

    # Display Volume Analysis
    print(f"\nVolume Data:")
    print(f"  Latest Volume: {human_readable(latest_volume)}")
    print(f"  Volume SMA ({VOLUME_SMA_PERIOD} period): {human_readable(latest_volume_sma)}")
    print(f"  Volume Signal: {volume_signal}")

    print(f"\nVotes -> Bullish: {bullish_votes}, Bearish: {bearish_votes}")
    print(f"Final Combined Trend: {final_trend}")
    print(f"Reason Summary: {reason_summary}")


# -----------------------
    # PLOT CHART (Optional)
    # -----------------------

    analysis_results = {
        "Coin": COIN_NAME,
        "coin_id": coin_id,
        "Price": human_readable(price),
        "EMA": human_readable(ema),
        "EMA_Signal": ema_signal,
        "RSI": human_readable(rsi),
        "RSI_Signal": rsi_signal,
        "ADX": human_readable(adx),
        "ADX_Signal": adx_filter,
        "MACD": human_readable(macd_value),
        "MACD_Signal": macd_filter,
        "Candlestick": candle_signal,
        "SMC": smc_signal,
        "Recent_High": human_readable(recent_high),
        "Recent_Low": human_readable(recent_low),
        "Fibonacci": ', '.join([f'{lvl}: {human_readable(val)}' for lvl, val in fib_levels.items()]),
        "Fibonacci_Signal": fib_signal if fib_signal else "No key Fib level",
        "Volume": human_readable(latest_volume),
        "Volume_SMA": human_readable(latest_volume_sma),
        "Volume_Signal": volume_signal,
        "Bullish_Votes": bullish_votes,
        "Bearish_Votes": bearish_votes,
        "Final_Trend": final_trend,
        "Reason_Summary": reason_summary
    }
    return analysis_results

def analyze_sentiment_async(coin_name, callback):
    """Run sentiment analysis in a separate thread and execute a callback when done."""
    future = executor.submit(reddit_sentiment_analysis, coin_name)
    future.add_done_callback(lambda f: callback(f.result()))

def reddit_sentiment_analysis(COIN_NAME):
    """Fetch Reddit posts and analyze sentiment."""
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                         client_secret=REDDIT_CLIENT_SECRET,
                         user_agent=REDDIT_USER_AGENT)

    try:
        subreddit = reddit.subreddit("CryptoCurrency")
        posts = subreddit.search(COIN_NAME, limit=50)  # Adjusted to 50 to reduce API load

        sid = SentimentIntensityAnalyzer()
        sentiment_sum = 0
        count = 0

        for post in posts:
            text = f"{post.title} {post.selftext}"
            scores = sid.polarity_scores(text)
            sentiment_sum += scores["compound"]
            count += 1

        avg_sentiment = sentiment_sum / count if count > 0 else 0

        if avg_sentiment >= 0.05:
            sentiment = "Strong"
        elif avg_sentiment <= -0.05:
            sentiment = "Weak"
        else:
            sentiment = "Average"

        # ✅ Clean formatted string output
        result = f"Sentiment: {sentiment}\nSentiment Score: {round(avg_sentiment, 2)}"
        return result

    except Exception as e:
        print(f"Error in Reddit Sentiment Analysis: {e}")
        return "Sentiment: Error\nSentiment Score: 0"

def fetch_coin_data():
    """Fetch cryptocurrency data from CoinMarketCap."""
    try:
        response = requests.get(CMC_URL, headers=CMC_HEADERS, params=CMC_PARAMS)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"HTTP Request failed: {e}")
        sys.exit()
    except ValueError:
        print("Error decoding JSON response from CoinMarketCap.")
        sys.exit()

    if "data" not in data:
        print("Error fetching data:", data.get("status", {}).get("error_message", "Unknown error"))
        sys.exit()

    return data["data"]

def filter_coins(coins):
    """Filter coins based on the specified criteria."""
    filtered = []
    for coin in coins:
        # Extract necessary fields with safe defaults
        rank = coin.get("cmc_rank")
        quote_usd = coin.get("quote", {}).get("USD", {})
        market_cap = quote_usd.get("market_cap")
        volume_change_24h = quote_usd.get("volume_change_24h", 0)
        percent_change_24h = quote_usd.get("percent_change_24h", 0)
        percent_change_7d = quote_usd.get("percent_change_7d", 0)
        percent_change_30d = quote_usd.get("percent_change_30d", 0)
        percent_change_60d = quote_usd.get("percent_change_60d", 0)
        percent_change_90d = quote_usd.get("percent_change_90d", 0)
        slug = coin.get("slug")  # Ensure 'slug' is available

        if (
                rank is not None and 1 <= rank <= 1000 and  # Adjusted to include top-ranked coins
                market_cap is not None and 1_000_000 <= market_cap <= 500_000_0000000 and
                slug.lower() not in ["usdt", "usdc", "dai"] and
                volume_change_24h > 2 and
                percent_change_7d > -3 and
                percent_change_30d > -10 and
                percent_change_60d > -20 and
                percent_change_90d > -30 and
                percent_change_24h > 2 and
                slug  # Ensure slug exists
        ):
            filtered.append(coin)
    return filtered

def sort_coins_by_90d_change(coins, top_n=TOP_N_COINS):
    """Sort coins based on 90-day percentage change and select top N."""
    sorted_coins = sorted(coins, key=lambda x: x['quote']['USD']['percent_change_90d'], reverse=True)
    return sorted_coins[:top_n]

def initialize_reddit_client():
    """Initialize the Reddit client using PRAW."""
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        # Test the Reddit client by accessing a read-only attribute
        reddit.read_only = True
    except Exception as e:
        print(f"Failed to initialize Reddit client: {e}")
        sys.exit()
    return reddit

def fetch_reddit_posts(reddit, coin_name, limit=SUBMISSIONS_LIMIT):
    """Fetch recent Reddit submissions related to the coin."""
    try:
        subreddit = reddit.subreddit("all")
        # Use more robust search terms to capture relevant discussions
        query = f'"{coin_name}" OR "{coin_name} crypto" OR "{coin_name}coin" OR "{coin_name} coin"'
        submissions = list(subreddit.search(query, limit=limit, sort='new'))
        return submissions
    except Exception as e:
        print(f"Error fetching Reddit posts for {coin_name}: {e}")
        return []

def determine_signal(ta_indicators, smc_analysis):
    """Determine trading signal based on technical indicators and SMC."""
    # Extract indicators
    rsi = ta_indicators.get('RSI')
    macd_line = ta_indicators.get('MACD_Line')
    signal_line = ta_indicators.get('Signal_Line')
    bb_upper = ta_indicators.get('BB_Upper')
    bb_lower = ta_indicators.get('BB_Lower')
    stochastic_k = ta_indicators.get('Stochastic_K')
    stochastic_d = ta_indicators.get('Stochastic_D')
    volume_trend = smc_analysis.get('Volume_Trend')
    current_price = smc_analysis.get('Current_Price', 0)

    signal_score = 0  # Initialize a score to aggregate signals

    # RSI Signals
    if rsi:
        if rsi < 30:
            signal_score += 1  # Bullish signal
        elif rsi > 70:
            signal_score -= 1  # Bearish signal

    # MACD Signals
    if macd_line is not None and not np.isnan(macd_line) and signal_line is not None and not np.isnan(signal_line):
        if macd_line > signal_line:
            signal_score += 1  # Bullish signal
        elif macd_line < signal_line:
            signal_score -= 1  # Bearish signal

    # Bollinger Bands Signals
    if bb_upper is not None and not np.isnan(bb_upper) and bb_lower is not None and not np.isnan(bb_lower):
        if current_price > bb_upper:
            signal_score -= 1  # Overbought
        elif current_price < bb_lower:
            signal_score += 1  # Oversold

    # Stochastic Oscillator Signals
    if stochastic_k is not None and not np.isnan(stochastic_k) and stochastic_d is not None and not np.isnan(stochastic_d):
        if stochastic_k > stochastic_d and stochastic_k > 80:
            signal_score -= 1  # Overbought
        elif stochastic_k < stochastic_d and stochastic_k < 20:
            signal_score += 1  # Oversold

    # Volume Trend Signals
    if volume_trend:
        if volume_trend == 'increasing':
            signal_score += 1  # Slight bullish
        elif volume_trend == 'decreasing':
            signal_score -= 1  # Slight bearish

    # Determine final signal based on aggregated score
    if signal_score >= 1.5:
        return 'Long'
    elif signal_score <= -1.5:
        return 'Short'
    else:
        return 'Neutral'

def analyze_sentiment(submissions):
    """
    Analyze sentiment of Reddit submissions using Afinn,
    returning the average sentiment score.
    """
    afinn = Afinn()  # You can also specify language if needed, e.g., Afinn(language='en')
    total_score = 0
    count = 0

    for submission in submissions:
        text = f"{submission.title} {submission.selftext}"
        score = afinn.score(text)
        total_score += score
        count += 1

    if count == 0:
        return 0  # Neutral if no submissions

    average_score = total_score / count
    return average_score

def get_historical_data(coingecko_id, days=365):
    """Fetch historical OHLC and volume data from CoinGecko."""
    try:
        # Fetch OHLC data
        ohlc_url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}/ohlc"
        ohlc_params = {'vs_currency': 'usd', 'days': days,'x_cg_demo_api_key': COINGECKO_API_KEY}
        ohlc_response = requests.get(ohlc_url, params=ohlc_params)
        ohlc_response.raise_for_status()
        ohlc_data = ohlc_response.json()

        if not ohlc_data:
            logging.warning(f"No OHLC data returned for {coingecko_id}.")
            return pd.DataFrame()

        df_ohlc = pd.DataFrame(ohlc_data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df_ohlc['timestamp'] = pd.to_datetime(df_ohlc['timestamp'], unit='ms')

        # Fetch Volume data
        market_chart_url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}/market_chart"
        market_chart_params = {'vs_currency': 'usd', 'days': days, 'interval': 'daily','x_cg_demo_api_key': COINGECKO_API_KEY}
        market_chart_response = requests.get(market_chart_url, params=market_chart_params)
        market_chart_response.raise_for_status()
        market_chart_data = market_chart_response.json()

        total_volumes = market_chart_data.get('total_volumes', [])

        if not total_volumes:
            logging.warning(f"No volume data returned for {coingecko_id}.")
            df_ohlc['volume'] = np.nan  # Assign NaN if volume data is unavailable
        else:
            df_volume = pd.DataFrame(total_volumes, columns=['timestamp', 'volume'])
            df_volume['timestamp'] = pd.to_datetime(df_volume['timestamp'], unit='ms')

            # Merge OHLC with Volume on timestamp
            df_merged = pd.merge(df_ohlc, df_volume, on='timestamp', how='left')
            df_merged['volume'] = df_merged['volume'].fillna(0.0)  # Handle missing volumes

            return df_merged

        return df_ohlc

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch historical data for {coingecko_id}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure

def apply_smart_money_concepts(ohlc_data):
    """Analyze volume trends or other Smart Money Concepts using OHLC data."""
    if ohlc_data.empty:
        logging.warning("No OHLC data available for Smart Money Concepts analysis.")
        return {}

    price_series = ohlc_data['close']

    # Check if 'volume' column exists
    if 'volume' in ohlc_data.columns:
        volumes = ohlc_data['volume']
    else:
        logging.warning("Volume data not available. Skipping volume trend analysis.")
        volumes = None

    if volumes is not None and len(volumes) >= 20:
        recent_avg = np.mean(volumes[-10:])
        previous_avg = np.mean(volumes[-20:-10])
        if recent_avg > previous_avg:
            volume_trend = 'increasing'
        elif recent_avg < previous_avg:
            volume_trend = 'decreasing'
        else:
            volume_trend = 'stable'
    else:
        volume_trend = 'N/A'

    # Current price
    current_price = price_series.iloc[-1] if not price_series.empty else 0.0

    return {
        'Volume': np.mean(volumes) if volumes is not None and not volumes.empty else 0.0,
        'MA20_Volume': np.mean(volumes[-20:]) if volumes is not None and len(volumes) >= 20 else 0.0,
        'Volume_Trend': volume_trend,
        'Current_Price': current_price
    }

def send_telegram_message(message):
    """Send a message to Telegram using the Bot API."""
    telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",  # Allows for formatting
        "disable_web_page_preview": True  # Allow link previews
    }
    try:
        response = requests.post(telegram_url, data=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to send Telegram message: {e}")

def format_market_cap(market_cap):
    """Format market cap with appropriate units (M, B, T)."""
    try:
        market_cap = float(market_cap)
    except (TypeError, ValueError):
        return "N/A"

    if market_cap >= 1_000_000_000_000:
        return f"{market_cap / 1_000_000_000_000:.2f}T"
    elif market_cap >= 1_000_000_000:
        return f"{market_cap / 1_000_000_000:.2f}B"
    elif market_cap >= 1_000_000:
        return f"{market_cap / 1_000_000:.2f}M"
    else:
        return f"${market_cap:,.2f}"

def perform_technical_analysis(ohlc_data):
    """Calculate technical indicators using OHLC data."""
    if ohlc_data.empty:
        logging.warning("No OHLC data available for technical analysis.")
        return {}



    price_series = ohlc_data['close']
    high_series = ohlc_data['high']
    low_series = ohlc_data['low']

    # Calculate Moving Averages
    sma30 = SMAIndicator(price_series, window=30).sma_indicator().iloc[-1] if len(price_series) >= 30 else None
    sma90 = SMAIndicator(price_series, window=90).sma_indicator().iloc[-1] if len(price_series) >= 90 else None

    # Calculate MACD
    if len(price_series) >= 26:
        macd = MACD(price_series, window_slow=26, window_fast=12, window_sign=9)
        macd_line = macd.macd().iloc[-1]
        signal_line = macd.macd_signal().iloc[-1]
        macd_histogram = macd.macd_diff().iloc[-1]
    else:
        macd_line = signal_line = macd_histogram = None

    # Calculate Bollinger Bands
    if len(price_series) >= 20:
        bollinger = BollingerBands(price_series, window=20, window_dev=2)
        bb_upper = bollinger.bollinger_hband().iloc[-1]
        bb_middle = bollinger.bollinger_mavg().iloc[-1]
        bb_lower = bollinger.bollinger_lband().iloc[-1]
    else:
        bb_upper = bb_middle = bb_lower = None

    # Calculate Stochastic Oscillator
    if len(price_series) >= 14 and len(high_series) >= 14 and len(low_series) >= 14:
        stochastic = StochasticOscillator(high=high_series, low=low_series, close=price_series, window=14,
                                          smooth_window=3)
        stochastic_k = stochastic.stoch().iloc[-1]
        stochastic_d = stochastic.stoch_signal().iloc[-1]
    else:
        stochastic_k = stochastic_d = None

    # Calculate RSI
    if len(price_series) >= 14:
        rsi = RSIIndicator(close=price_series, window=14).rsi().iloc[-1]
    else:
        rsi = None

    # Calculate Pivot Points
    pivot_points = calculate_pivot_points(ohlc_data)

    return {
        'SMA30': sma30,
        'SMA90': sma90,
        'MACD_Line': macd_line,
        'Signal_Line': signal_line,
        'MACD_Histogram': macd_histogram,
        'BB_Upper': bb_upper,
        'BB_Middle': bb_middle,
        'BB_Lower': bb_lower,
        'Stochastic_K': stochastic_k,
        'Stochastic_D': stochastic_d,
        'RSI': rsi,
        'Pivot_Points': pivot_points
    }

def format_percentage(percent):
    """Format percentage with sign and limited decimal places."""
    try:
        percent = float(percent)
    except (TypeError, ValueError):
        return "N/A"
    sign = "+" if percent > 0 else ""
    return f"{sign}{percent:.1f}%"

def display_filtered_coins(filtered_coins):
    """Display the list of filtered coins."""
    print(f"Found {len(filtered_coins)} coins matching all criteria.\n")
    for coin in filtered_coins:
        name = coin.get('name', 'N/A')
        symbol = coin.get('symbol', 'N/A')
        rank = coin.get('cmc_rank', 'N/A')
        quote = coin.get("quote", {}).get("USD", {})
        market_cap = quote.get('market_cap', 'N/A')
        volume_change_24h = quote.get('volume_change_24h', 'N/A')
        percent_change_7d = quote.get('percent_change_7d', 'N/A')
        percent_change_90d = quote.get('percent_change_90d', 'N/A')

        print(
            f"Name: {name} | Symbol: {symbol} | Rank: {rank} | "
            f"Market Cap: {format_market_cap(market_cap)} | 7d% Change: {format_percentage(percent_change_7d)} | "
            f"90d% Change: {format_percentage(percent_change_90d)} | 24h Volume Change: {format_percentage(volume_change_24h)}"
        )
    print("\n")

def fetch_and_analyze(reddit, coin):
    """Fetch Reddit posts and analyze sentiment for a single coin."""
    name = coin.get('name', 'N/A')
    symbol = coin.get('symbol', 'N/A')
    rank = coin.get('cmc_rank', 'N/A')
    slug = coin.get('slug', '')
    submissions = fetch_reddit_posts(reddit, name)
    sentiment_score = analyze_sentiment(submissions)

    # Safely extract 'price' from 'quote'->'USD'
    quote_usd = coin.get('quote', {}).get('USD', {})
    price = quote_usd.get('price', None)
    market_cap = quote_usd.get('market_cap', 'N/A')
    percent_change_90d = quote_usd.get('percent_change_90d', 'N/A')

    return {
        'name': name,
        'symbol': symbol,
        'sentiment': sentiment_score,
        'price': price,
        'market_cap': market_cap,
        'percent_change_90d': percent_change_90d,
        'cmc_rank': rank,
        'slug': slug
    }

def get_coingecko_coin_map():
    """Fetch and map CoinGecko symbols to their respective IDs."""
    url = "https://api.coingecko.com/api/v3/coins/list"
    try:
        response = requests.get(url)
        response.raise_for_status()
        coins = response.json()
        symbol_to_ids = defaultdict(list)
        for coin in coins:
            symbol = coin['symbol'].upper()
            symbol_to_ids[symbol].append(coin['id'])
        return symbol_to_ids
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch CoinGecko coins list: {e}")
        return defaultdict(list)

def choose_best_coin_id(symbol_ids):
    """Choose the most appropriate CoinGecko ID when multiple IDs share the same symbol."""
    # Implement selection logic, e.g., choose the one with the highest market cap
    # For simplicity, return the first ID
    return symbol_ids[0] if symbol_ids else None

def parse_price(price_str):

    try:
        return float(price_str.replace('$', ''))
    except (ValueError, AttributeError):
        logging.error(f"Invalid price format: {price_str}")
        return None

def determine_entry_exit_points(
    current_price: float,
    signal: str,
    pivot_points: dict,
    moving_averages: dict,
    percent_band: float = 0.05  # ±5% for 'Neutral'
) -> dict:
    """
    Produce up to 3 entry and 3 exit points based on the current price,
    trading signal (Long/Short/Neutral), pivot points, and MAs.
    """

    def valid_level(val):
        """Check that val is not None, not NaN, and > 0."""
        return (val is not None
                and isinstance(val, (float, int))
                and not math.isnan(val)
                and val > 0)

    # 1) Gather pivot points
    candidate_levels = []
    for lvl in ["Pivot", "Support1", "Support2", "Resistance1", "Resistance2"]:
        val = pivot_points.get(lvl)
        if valid_level(val):
            candidate_levels.append(val)

    # 2) Gather moving averages
    for ma_key in ["SMA30", "SMA90"]:  # or SMA50, SMA200, etc.
        val = moving_averages.get(ma_key)
        if valid_level(val):
            candidate_levels.append(val)

    # Remove duplicates, sort ascending
    candidate_levels = sorted(set(candidate_levels))

    # Basic validation
    if not valid_level(current_price):
        logging.warning("Invalid current price. Returning empty points.")
        return {"Entry_Points": [], "Exit_Points": []}

    # Partition logic based on signal
    if signal == "Long":
        entries = [lvl for lvl in candidate_levels if lvl < current_price]
        exits = [lvl for lvl in candidate_levels if lvl > current_price]

        # Sort so that entries are descending (closest below price first),
        # exits ascending (closest above price first)
        entries.sort(key=lambda x: abs(current_price - x), reverse=True)
        exits.sort(key=lambda x: abs(current_price - x))

    elif signal == "Short":
        entries = [lvl for lvl in candidate_levels if lvl > current_price]
        exits = [lvl for lvl in candidate_levels if lvl < current_price]

        # Entries ascending, exits descending
        entries.sort(key=lambda x: abs(current_price - x))
        exits.sort(key=lambda x: abs(current_price - x), reverse=True)

    else:
        # NEUTRAL signal
        # We pick levels within ±percent_band around current_price
        lower_bound = current_price * (1 - percent_band)
        upper_bound = current_price * (1 + percent_band)

        # "Entries" = levels below or equal to current_price, but above lower_bound
        entries = [lvl for lvl in candidate_levels
                   if lower_bound <= lvl <= current_price]
        # "Exits" = levels above or equal to current_price, but below upper_bound
        exits = [lvl for lvl in candidate_levels
                 if current_price <= lvl <= upper_bound]

        # Sort them so entries are descending from current_price down,
        # exits ascending away from current_price
        entries.sort(key=lambda x: abs(current_price - x), reverse=True)
        exits.sort(key=lambda x: abs(current_price - x))

    # Take up to 3 from each side
    entries = entries[:3]
    exits = exits[:3]

    # --- FALLBACK LOGIC (in case we have fewer than 3 real levels) ---
    def add_fallback_points(points, needed, base_price, direction="below", step=0.02):
        """
        Add fallback points in 2% increments (or whichever step you choose).
        `direction="below"` => fallback < base_price
        `direction="above"` => fallback > base_price
        """
        attempt = 1
        while len(points) < needed and attempt < 20:
            if direction == "below":
                fb = base_price * (1 - step * attempt)
                # Only add if > 0, not a duplicate, and truly below base_price
                if fb > 0 and fb < base_price and fb not in points:
                    points.append(fb)
            else:  # "above"
                fb = base_price * (1 + step * attempt)
                if fb > base_price and fb not in points:
                    points.append(fb)
            attempt += 1
        return points

    # For LONG/SHORT, you likely already have fallback logic, but let's do it for NEUTRAL as well:
    if signal == "Neutral":
        # We'll define "entries" as fallback below current_price if we don't have enough
        if len(entries) < 3:
            entries = add_fallback_points(entries, 3, current_price, direction="below", step=0.01)
        # We'll define "exits" as fallback above current_price
        if len(exits) < 3:
            exits = add_fallback_points(exits, 3, current_price, direction="above", step=0.01)

        # Re-sort them
        entries.sort(key=lambda x: abs(current_price - x), reverse=True)
        exits.sort(key=lambda x: abs(current_price - x))
        entries = entries[:3]
        exits = exits[:3]

    # Similarly, if you want fallback for Long or Short, replicate the pattern:
    elif signal == "Long":
        if len(entries) < 3:
            entries = add_fallback_points(entries, 3, current_price, "below", 0.02)
            entries.sort(key=lambda x: abs(current_price - x), reverse=True)
            entries = entries[:3]
        if len(exits) < 3:
            exits = add_fallback_points(exits, 3, current_price, "above", 0.02)
            exits.sort(key=lambda x: abs(current_price - x))
            exits = exits[:3]

    elif signal == "Short":
        if len(entries) < 3:
            entries = add_fallback_points(entries, 3, current_price, "above", 0.02)
            entries.sort(key=lambda x: abs(current_price - x))
            entries = entries[:3]
        if len(exits) < 3:
            exits = add_fallback_points(exits, 3, current_price, "below", 0.02)
            exits.sort(key=lambda x: abs(current_price - x), reverse=True)
            exits = exits[:3]

    # --- Final Formatting ---
    entry_points = [f"${lvl:.4f}" for lvl in entries]
    exit_points = [f"${lvl:.4f}" for lvl in exits]

    # Log results for debugging
    logging.debug(f"{signal} signal => Entry Points: {entry_points}")
    logging.debug(f"{signal} signal => Exit Points: {exit_points}")

    return {
        "Entry_Points": entry_points,
        "Exit_Points": exit_points
    }

def format_indicator(value, is_currency=False):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    if is_currency:
        return f"${value:.2f}"
    return f"{value:.2f}"

def calculate_pivot_points(df):
    if df.empty or not {'high', 'low', 'close'}.issubset(df.columns):
        logging.warning("Insufficient data to calculate Pivot Points.")
        return {}

    # Using the last available day's data
    last = df.iloc[-1]
    high = last['high']
    low = last['low']
    close = last['close']

    pivot = (high + low + close) / 3
    resistance1 = (2 * pivot) - low
    support1 = (2 * pivot) - high
    resistance2 = pivot + (high - low)
    support2 = pivot - (high - low)

    # Validate that support and resistance levels are positive and within a reasonable range
    validated_pivot_points = {}
    validated_pivot_points['Pivot'] = pivot if pivot > 0 and pivot < high * 2 else None
    validated_pivot_points['Support1'] = support1 if support1 > 0 and support1 < pivot else None
    validated_pivot_points['Support2'] = support2 if support2 > 0 and support2 < support1 else None
    validated_pivot_points['Resistance1'] = resistance1 if resistance1 > pivot and resistance1 < high * 2 else None
    validated_pivot_points[
        'Resistance2'] = resistance2 if resistance2 > resistance1 and resistance2 < high * 2 else None

    # Log invalid levels
    for level, value in validated_pivot_points.items():
        if value is None:
            logging.warning(
                f"{level} level is invalid (value: {value}). It will be excluded from entry/exit point calculations.")
        else:
            logging.debug(f"{level} level: {value}")

    return validated_pivot_points

def calculate_moving_averages(df):
    if df.empty or 'close' not in df.columns:
        logging.warning("Insufficient data to calculate Moving Averages.")
        return {}

    sma30 = SMAIndicator(df['close'], window=30).sma_indicator().iloc[-1] if len(df) >= 30 else None
    sma90 = SMAIndicator(df['close'], window=90).sma_indicator().iloc[-1] if len(df) >= 90 else None

    sma30 = sma30 if sma30 and sma30 > 0 else None
    sma90 = sma90 if sma90 and sma90 > 0 else None

    if sma30:
        logging.debug(f"SMA50: {sma30}")
    else:
        logging.warning("SMA50 is not available.")

    if sma90:
        logging.debug(f"SMA200: {sma90}")
    else:
        logging.warning("SMA200 is not available.")

    return {
        'SMA30': sma30,
        'SMA90': sma90
    }

def clean_symbol(symbol):
    """
    Cleans the symbol by removing special characters and converting to uppercase.
    E.g., '$PURPE' becomes 'PURPE'
    """
    # Remove any character that is not alphanumeric
    cleaned = re.sub(r'[^A-Z0-9]', '', symbol.upper())
    return cleaned

def format_current_price(price: float) -> str:

    # Handle sign
    is_negative = (price < 0)
    abs_price = abs(price)

    if abs_price >= 1:
        # Truncate to 2 decimals
        truncated = math.floor(abs_price * 10 ** 2) / 10 ** 2
        format_str = f"{truncated:.2f}"
    elif abs_price >= 0.0001:
        # Truncate to 6 decimals
        truncated = math.floor(abs_price * 10 ** 6) / 10 ** 6
        format_str = f"{truncated:.6f}"
    else:
        # Truncate to 8 decimals
        truncated = math.floor(abs_price * 10 ** 8) / 10 ** 8
        format_str = f"{truncated:.8f}"

    # Strip trailing zeros and unnecessary decimal point
    format_str = format_str.rstrip('0').rstrip('.')

    # Reapply negative sign if needed
    if is_negative:
        format_str = "-" + format_str

    return format_str

def mainfoo(update_progress_callback):
    """
    Modified main() that reports progress via `update_progress_callback(percentage)`.
    We increment progress at key steps, plus inside concurrency loops.
    """
    # Constants (Adjust if needed)
    SMC_WEIGHT = 2.0  # Increase this to emphasize "smart money" influence
    current_step = 0
    major_steps_without_loops = 8

    total_steps = major_steps_without_loops

    # We'll add concurrency steps after we know how many coins we have in top_n_coins and top_coins_candidates.

    def step_done():
        """Increment the current_step, call the progress callback."""
        nonlocal current_step, total_steps
        current_step += 1
        # Ensure we don't exceed total_steps
        progress = min((current_step / total_steps) * 100, 100)
        update_progress_callback(progress)

    # -------------------------------------------------------------------------
    # Step 1: Fetch and Filter Coins from CoinMarketCap
    # -------------------------------------------------------------------------
    coins = fetch_coin_data()
    if not coins:
        logging.error("No coins fetched from CoinMarketCap.")
        return "[color=ff0000]No coins fetched from CoinMarketCap.[/color]"
    filtered_coins = filter_coins(coins)
    step_done()

    # -------------------------------------------------------------------------
    # Step 1.1: Display Filtered Coins
    # -------------------------------------------------------------------------
    display_filtered_coins(filtered_coins)
    step_done()

    # -------------------------------------------------------------------------
    # Step 2: Sort and Select Top N Coins Based on 90-Day Percentage Change
    # -------------------------------------------------------------------------
    top_n_coins = sort_coins_by_90d_change(filtered_coins, top_n=TOP_N_COINS)
    logging.info(f"Selected Top {len(top_n_coins)} coins based on 90-day performance.\n")
    step_done()

    # -------------------------------------------------------------------------
    # Step 3: Initialize Reddit Client
    # -------------------------------------------------------------------------
    reddit = initialize_reddit_client()
    if not reddit:
        logging.error("Failed to initialize Reddit client.")
        return "[color=ff0000]Failed to initialize Reddit client.[/color]"
    step_done()

    # For concurrency in step #4, we will have len(top_n_coins) sub-steps
    # For concurrency in step #8, we will have up to 10 sub-steps
    # So total_steps = major_steps_without_loops + len(top_n_coins) + (up to 10)
    # We'll finalize after we know how many coins are left for step #8.

    coin_sentiments = []
    # We'll do step #4 concurrency after we define total_steps, so let's hold off until we see top_n_coins.
    # But we actually need top_n_coins for concurrency, so let's finalize total_steps after we do step #2 but before step #4.

    # We don't yet know how many coins will go to technical analysis, but the max is 10.
    # We can assume up to 10. So let's define total_steps = major_steps_without_loops + len(top_n_coins) + 10
    total_steps = major_steps_without_loops + len(top_n_coins) + 10

    # -------------------------------------------------------------------------
    # Step 4: Fetch Sentiment for Each Coin Concurrently
    # -------------------------------------------------------------------------
    logging.info("Fetching sentiments concurrently...\n")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_coin = {executor.submit(fetch_and_analyze, reddit, coin): coin for coin in top_n_coins}
        for future in as_completed(future_to_coin):
            coin = future_to_coin[future]
            try:
                result = future.result()
                coin_sentiments.append(result)
                logging.info(
                    f"Processed Sentiment for {result['name']} ({result['symbol']}): {result['sentiment']:.4f}")
            except Exception as e:
                logging.error(f"Error processing {coin.get('name', 'N/A')}: {e}", exc_info=True)
            finally:
                # Each coin in concurrency is 1 step
                step_done()

    if not coin_sentiments:
        logging.error("No sentiment data processed.")
        return "[color=ff0000]No sentiment data processed.[/color]"

    logging.info("\nAll sentiments fetched and analyzed.\n")

    # -------------------------------------------------------------------------
    # Step 5: Compute Composite Score and Sort Coins
    # -------------------------------------------------------------------------
    for coin in coin_sentiments:
        try:
            normalized_performance = float(coin['percent_change_90d']) / 1000
        except (TypeError, ValueError):
            normalized_performance = 0

        try:
            normalized_sentiment = (float(coin['sentiment']) + 1) / 2
        except (TypeError, ValueError):
            normalized_sentiment = 0

        composite_score = (PERFORMANCE_WEIGHT * normalized_performance) + (SENTIMENT_WEIGHT * normalized_sentiment)
        coin['composite_score'] = composite_score

    final_sorted = sorted(
        coin_sentiments,
        key=lambda x: x['composite_score'],
        reverse=True
    )
    step_done()

    # -------------------------------------------------------------------------
    # Step 6: Select Final Top N Coins
    # -------------------------------------------------------------------------
    final_top = final_sorted[:FINAL_TOP_N]
    # Sort final top N by rank (ascending)
    final_top_sorted = sorted(final_top, key=lambda x: x['cmc_rank'])
    logging.info(f"Final Top {len(final_top_sorted)} Coins Based on Sentiment and Performance:\n")
    step_done()

    # -------------------------------------------------------------------------
    # Step 7: Map CoinMarketCap Symbols to CoinGecko IDs
    # -------------------------------------------------------------------------
    symbol_to_ids = get_coingecko_coin_map()
    if not symbol_to_ids:
        logging.error("No CoinGecko ID mappings found. Skipping Technical Analysis.")
        return "[color=ff0000]No CoinGecko ID mappings found.[/color]"
    step_done()

    # -------------------------------------------------------------------------
    # Step 8: Select Up to 10 Coins for Technical Analysis
    # -------------------------------------------------------------------------
    top_coins_candidates = final_top_sorted[:TECHNICAL_ANALYSIS_CANDIDATES]

    ta_messages = []
    skipped_coins = []
    analyzed_coins = 0
    analyzed_coins_data = []

    for coin in top_coins_candidates:
        try:
            name = coin['name']
            symbol = coin['symbol'].upper()
            slug = coin.get('slug', '')
            cmc_url = f"https://coinmarketcap.com/currencies/{slug}/" if slug else "N/A"
            logging.info(f"\nProcessing Technical Analysis for {name} ({symbol})")

            cleaned_symbol = clean_symbol(symbol)
            logging.debug(f"Cleaned symbol for mapping: {cleaned_symbol}")

            if cleaned_symbol in symbol_to_ids:
                if len(symbol_to_ids[cleaned_symbol]) == 1:
                    selected_coin_id = symbol_to_ids[cleaned_symbol][0]
                else:
                    selected_coin_id = choose_best_coin_id(symbol_to_ids[cleaned_symbol])
            else:
                selected_coin_id = None

            if not selected_coin_id:
                logging.warning(f"Could not find CoinGecko ID for {symbol}. Skipping Technical Analysis.")
                skipped_coins.append({
                    'name': name,
                    'symbol': symbol,
                    'cmc_url': cmc_url,
                    'reason': 'Missing CoinGecko ID'
                })
                # Step done for this coin
                step_done()
                continue

            historical_data = get_historical_data(selected_coin_id, days=365)
            if historical_data.empty:
                logging.warning(f"No historical data available for {name}. Skipping Technical Analysis.")
                skipped_coins.append({
                    'name': name,
                    'symbol': symbol,
                    'cmc_url': cmc_url,
                    'reason': 'Insufficient historical data'
                })
                step_done()
                continue

            ta_indicators = perform_technical_analysis(historical_data)
            logging.info(f"Technical Analysis Indicators for {name}: {ta_indicators}")

            smc_analysis = apply_smart_money_concepts(historical_data)
            logging.info(f"Smart Money Concepts Analysis for {name}: {smc_analysis}")

            # Capture an SMC score from the analysis if present
            coin['smc_score'] = smc_analysis.get('SMC_Score', 0.0)

            signal = determine_signal(ta_indicators, smc_analysis)
            logging.info(f"Trading Signal for {name}: {signal}")

            current_price = coin.get("price")
            logging.debug(f"Raw current_price for {name} ({symbol}): {current_price}")
            if current_price is None:
                current_price = '0.00'
            try:
                current_price = float(current_price)
            except (TypeError, ValueError):
                current_price = 0.0

            logging.debug(f"Converted current_price for {name} ({symbol}): {current_price} USD")

            pivot_points = ta_indicators.get('Pivot_Points', {})
            moving_averages = ta_indicators.get('Moving_Averages', {})

            entry_exit_points = determine_entry_exit_points(
                current_price=current_price,
                signal=signal,
                pivot_points=pivot_points,
                moving_averages=moving_averages
            )
            logging.info(f"Entry Points: {entry_exit_points['Entry_Points']}")
            logging.info(f"Exit Points: {entry_exit_points['Exit_Points']}")

            if not entry_exit_points['Entry_Points'] or not entry_exit_points['Exit_Points']:
                logging.warning(f"No valid Entry/Exit Points for {name}. Skipping adding to Telegram message.")
                skipped_coins.append({
                    'name': name,
                    'symbol': symbol,
                    'cmc_url': cmc_url,
                    'reason': 'Invalid Entry/Exit Points'
                })
                step_done()
                continue

            formatted_price = format_current_price(current_price)
            sma30 = ta_indicators.get("SMA30", 0.0)
            sma90 = ta_indicators.get("SMA90", 0.0)
            macd_hist = ta_indicators.get("MACD_Histogram", 0.0)
            rsi = ta_indicators.get("RSI", 50.0)
            signal_emoji = {
                "Long": "🔼",
                "Short": "🔻",
                "Neutral": "⚖️"
            }
            signal_emoji_symbol = signal_emoji.get(signal, "❓")

            # ---- Improved TA Scoring Logic ----
            coin['ta_score'] = 0.0
            # (1) Weight the trading signal
            if signal == "Long":
                coin['ta_score'] += 2.0
            elif signal == "Short":
                coin['ta_score'] -= 2.0

            # (2) Compare SMA30 vs. SMA90
            if isinstance(sma30, (float, int)) and isinstance(sma90, (float, int)):
                if sma30 > sma90:
                    coin['ta_score'] += 1.0
                elif sma30 < sma90:
                    coin['ta_score'] -= 1.0

            # (3) RSI-based scoring
            if isinstance(rsi, (float, int)):
                if rsi < 30:
                    coin['ta_score'] += 1.5
                elif 30 <= rsi < 50:
                    coin['ta_score'] += 0.5
                elif 70 < rsi <= 80:
                    coin['ta_score'] -= 0.5
                elif rsi > 80:
                    coin['ta_score'] -= 1.5

            # (4) MACD histogram
            if isinstance(macd_hist, (float, int)):
                if macd_hist > 0:
                    coin['ta_score'] += 0.5
                else:
                    coin['ta_score'] -= 0.5
            # -----------------------------------

            ta_message = (
                f"\n\n📊 *Technical Analysis for {name} ({symbol})*\n"
                f"🎖️ **Rank:** {coin['cmc_rank']}\n"
                f"💲 **Current Price:** {formatted_price}\n"
                f"💰 **Market Cap:** {format_market_cap(coin.get('market_cap'))}\n"
                f"🎭 **Sentiment Score:** {coin['sentiment']:.2f}\n"
                f"📈 **Performance Score:** {coin['composite_score']:.2f}\n"
                f"🔸 **30-Day SMA:** {format_indicator(sma30, is_currency=True)}\n"
                f"🔸 **90-Day SMA:** {format_indicator(sma90, is_currency=True)}\n"
                f"👉 **RSI:** {format_indicator(rsi)}\n"
                f"🔸 **MACD Line:** {format_indicator(ta_indicators.get('MACD_Line'))}\n"
                f"🔸 **Signal Line:** {format_indicator(ta_indicators.get('Signal_Line'))}\n"
                f"🔸 **MACD Histogram:** {format_indicator(macd_hist)}\n"
                f"📶 **Stochastic Oscillator:** %K - {format_indicator(ta_indicators.get('Stochastic_K'))}, "
                f"%D - {format_indicator(ta_indicators.get('Stochastic_D'))}\n"
                f"🦈 **Volume Trend:** {smc_analysis.get('Volume_Trend', 'N/A').capitalize()}\n"
                f"👀 **Bollinger Bands:** Upper - {format_indicator(ta_indicators.get('BB_Upper'), is_currency=True)}, "
                f"Middle - {format_indicator(ta_indicators.get('BB_Middle'), is_currency=True)}, "
                f"Lower - {format_indicator(ta_indicators.get('BB_Lower'), is_currency=True)}\n"
                f"🤑 **Trading Signal:** {signal_emoji_symbol} **{signal.upper()}**\n"
                f"🚀 **Entry Points:** {', '.join(entry_exit_points['Entry_Points'])}\n"
                f"⚠️ **Exit Points:** {', '.join(entry_exit_points['Exit_Points'])}\n"
                f"🌐 [ View on CoinMarketCap]({cmc_url})"
            )
            ta_messages.append(ta_message)

            coin['ta_message'] = ta_message
            analyzed_coins += 1
            analyzed_coins_data.append(coin)

        except Exception as e:
            slug = coin.get('slug', '')
            cmc_url = f"https://coinmarketcap.com/currencies/{slug}/" if slug else "N/A"
            logging.error(f"An error occurred while processing {coin.get('name')} ({coin.get('symbol')}): {e}",
                          exc_info=True)
            skipped_coins.append({
                'name': coin.get('name', 'Unknown'),
                'symbol': coin.get('symbol', 'Unknown'),
                'cmc_url': cmc_url,
                'reason': str(e)
            })
        finally:
            # Each coin in step #8 is 1 step
            step_done()

    if not ta_messages:
        logging.error("No sentiment or technical analysis messages were generated.")
        return "[color=ff0000]No sentiment or technical analysis messages were generated.[/color]"

    logging.info("\nAll sentiments fetched and analyzed.\n")

    # -------------------------------------------------------------------------
    # Step 9: Incorporate SMC weighting, then prepare the final top 3 coins
    # -------------------------------------------------------------------------
    for coin in analyzed_coins_data:
        smc_score = coin.get('smc_score', 0.0)
        coin['final_score'] = coin['ta_score'] + (smc_score * SMC_WEIGHT)

    # Sort by combined TA + SMC score
    sorted_by_final = sorted(analyzed_coins_data, key=lambda c: c.get('final_score', 0), reverse=True)
    top_3_ta_coins = sorted_by_final[:3]

    # Rebuild messages for top 3 only
    ta_messages_top_3 = [coin['ta_message'] for coin in top_3_ta_coins if 'ta_message' in coin]

    if skipped_coins:
        skipped_lines = []
        for skipped in skipped_coins:
            skipped_lines.append(
                f"❌ *{skipped['name']} ({skipped['symbol']})* skipped: {skipped['reason']}.\n"
                f"[🌐 View on CoinMarketCap]({skipped['cmc_url']})"
            )
        skipped_message = "\n".join(skipped_lines)
    else:
        skipped_message = "✅ *All selected coins* were successfully analyzed."

    if ta_messages_top_3:
        comprehensive_ta_message = "\n".join(ta_messages_top_3)
    else:
        comprehensive_ta_message = (
            "⚠️ *No coins were successfully analyzed for Technical Analysis due to insufficient data.*"
        )

    full_message = (
        f"🎯 *Technical Analysis Results*\n\n{comprehensive_ta_message}\n\n{skipped_message}"
    )

    send_telegram_message(full_message)
    logging.info("\nComprehensive alert with top 3 TA coins sent to Telegram successfully.")
    return full_message


BASE_URL = "https://api.coingecko.com/api/v3"
DEFAULT_DAYS = 180  # e.g. 180 days, hopefully enough for MACD
# If you have a free public usage, keep the above. If you have a Pro key, change domain and add header.



SUBREDDITS = ["CryptoCurrency", "cryptocurrency"]

class LoadingDialog(MDBoxLayout):
    """Dialog content: label + progress bar."""

class ResultsDialogContent(MDBoxLayout):
    """Dialog content: scrollable label for final results."""
    message = StringProperty("")
#Window.size = (310, 630)
RAPIDAPI_KEY = ""  # Replace with your actual RapidAPI key
HEADERS = {
    "X-RapidAPI-Host": "coingecko.p.rapidapi.com",
    "X-RapidAPI-Key": RAPIDAPI_KEY
}
# Firebase Configuration
firebase_config = {
    "apiKey": "",
    "authDomain": "",
    "databaseURL": "",
    "projectId": "",
    "storageBucket": "",
    "messagingSenderId": "",
    "appId": "",
    "measurementId": ""
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
db = firebase.database()

def get_error_message(error):
    """Map Firebase and network errors to user-friendly messages."""
    if "EMAIL_NOT_FOUND" in str(error):
        return "Email not found. Please check your email."
    elif "INVALID_PASSWORD" in str(error):
        return "Invalid password. Please try again."
    elif "EMAIL_EXISTS" in str(error):
        return "Email already in use. Try logging in."
    elif "NETWORK_ERROR" in str(error):
        return "Network error. Please check your connection."
    else:
        return "Please fill correct email & password."
class SafeAsyncImage(AsyncImage):
    def on_texture(self, instance, value):
        """Handle successful texture loading."""
        if not value:
            Logger.error("SafeAsyncImage: Failed to load texture, displaying placeholder.")
            self.source = "cph.png"  # Local placeholder image
        super().on_texture(instance, value)

    def on_error(self, *args):
        """Handle image loading errors."""
        Logger.error(f"SafeAsyncImage: Error loading image from {self.source}")
        self.source = "cph.png"

def format_large_number(value):
    """Format large numbers into k, m, b, t."""
    try:
        num = float(value)
        if num >= 1_000_000_000_000:
            return f"{num / 1_000_000_000_000:.2f}t"
        elif num >= 1_000_000_000:
            return f"{num / 1_000_000_000:.2f}b"
        elif num >= 1_000_000:
            return f"{num / 1_000_000:.2f}m"
        elif num >= 1_000:
            return f"{num / 1_000:.2f}k"
        else:
            return f"{num:.2f}"
    except (ValueError, TypeError):
        return "N/A"
class TokenCard(MDCard):
    def __init__(self, token, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.padding = dp(10)
        self.spacing = dp(10)
        self.size_hint_y = None
        self.height = dp(600)  # Increased height to accommodate all sections
        self.md_bg_color = [1, 1, 1, 1]
        self.radius = [12, 12, 12, 12]
        self.elevation = 10

        # Token image section
        image = SafeAsyncImage(
            source=token['image_url'],
            size_hint=(1, None),
            height=dp(160),
            allow_stretch=True,
            keep_ratio=True
        )
        self.add_widget(image)
        market_cap_value = float(
            token['market_cap'].replace('k', '').replace('m', '').replace('b', '').replace('t', ''))
        market_cap_color = self.get_market_cap_color(market_cap_value)
        # Token details section
        details = MDLabel(
            text=(
                f"[b]Name:[/b] {token['name']} ({token['symbol']})\n"
                f"[b]Chain:[/b] {token['chain']}\n"
                f"[b]Price (USD):[/b] ${token['price']}\n"
                f"[b]Market Cap (USD):[/b] [color={market_cap_color}]${token['market_cap']}[/color]\n"               
                f"[b]Volume (24h, USD):[/b] ${token['volume']}\n"                
                f"[b]Price Change (24h):[/b] {token['price_change_24h']}%\n"
                f"[b]Liquidity (USD):[/b] ${token['liquidity']}\n"
                f"[b]Pair Created At:[/b] {token['pair_created_at']}"
            ),
            markup=True,
            size_hint_y=None,
            height=dp(180),
            theme_text_color="Custom",
            text_color=[0, 0, 0, 1],
            valign="top",
            halign="left"
        )
        self.add_widget(details)


        # Links section (vertically stacked buttons)
        link_buttons = BoxLayout(
            orientation="vertical",
            spacing=dp(8),
            size_hint=(1, None),
            height=dp(200),
            padding=[dp(10), dp(10)]
        )

        dex_button = MDRaisedButton(
            text="DexScreener",
            #style= "elevated",
            size_hint=(1, None),
            height=dp(40),
            on_release=lambda x: self.open_link(token['url'])
        )
        website_button = MDRaisedButton(
            text="Website",
            #style="elevated",
            size_hint=(1, None),
            height=dp(40),
            on_release=lambda x: self.open_link(token['website']),
            disabled=token['website'] == "No website available"
        )
        twitter_button = MDRaisedButton(
            text="Twitter",
            #style="elevated",
            size_hint=(1, None),
            height=dp(40),
            on_release=lambda x: self.open_link(token['twitter']),
            disabled=token['twitter'] == "No Twitter link available"
        )
        telegram_button = MDRaisedButton(
            text="Telegram",
            #style="elevated",
            size_hint=(1, None),
            height=dp(40),
            on_release=lambda x: self.open_link(token['telegram']),
            disabled=token['telegram'] == "No Telegram link available"
        )

        link_buttons.add_widget(dex_button)
        link_buttons.add_widget(website_button)
        link_buttons.add_widget(twitter_button)
        link_buttons.add_widget(telegram_button)
        self.add_widget(link_buttons)

    def open_link(self, url):
        """Open the link in a web browser."""
        if url and "No " not in url:  # Ensure the link is valid
            import webbrowser
            webbrowser.open(url)

    def get_market_cap_color(self, value):
        """Return a color code based on market cap value."""
        if value >= 500:  # Highlight for market cap above 1 billion
            return "0000ff"# blue for very high market cap
        elif value >= 250:  # Highlight for market cap above 100 million
            return "00ff00"  # Green for medium market cap
        else:
              # red for low market cap
            return "ff0000"

# --- 1) GET MARKET DATA ---
def fetch_market_data():
    """
    Fetch real-time market data for top coins from CoinMarketCap.
    Returns a list of coin data dicts or an empty list if an error occurs.
    """

    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": "",  # Replace with valid API key
    }
    params = {
        "start": 250,
        "limit": 1250,  # Adjust to fetch more or fewer coins
        "convert": "USD"
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        data = response.json().get("data", [])
        return data
    except requests.RequestException as e:
        print(f"[Error] Unable to fetch market data from CoinMarketCap: {e}")
        return []

# --- NUMBER ABBREVIATION HELPER (OPTIONAL) ---
def abbreviate_number(num):
    """
    Convert large numbers to a string with K, M, B, or T suffix.
    """
    abs_num = abs(num)
    if abs_num >= 1_000_000_000_000:
        return f"{num / 1_000_000_000_000:.2f}T"
    elif abs_num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif abs_num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif abs_num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return f"{num:.2f}"

# --- 5) FETCH SENTIMENT (CALLED ONLY FOR SHORT-LIST) ---
def fetch_public_sentiment(coin_name):
    """
    Fetch public sentiment for a coin from Reddit using simple keyword matching.
    Returns an integer sentiment score between 0 and 10.
    """
    search_query = coin_name.replace(" ", "+")
    url = f"https://www.reddit.com/search.json?q={search_query}&limit=10"
    headers = {"User-Agent": "crypto-recommendation-bot"}

    # Simple keyword sets (expand as needed)
    positive_keywords = ["buy", "bullish", "hold", "moon"]
    negative_keywords = ["sell", "bearish", "dump", "scam"]

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        posts = response.json().get("data", {}).get("children", [])

        sentiment_score = 0
        for post in posts:
            post_data = post.get("data", {})
            title = post_data.get("title", "").lower()
            self_text = post_data.get("selftext", "").lower()
            content = f"{title} {self_text}"

            # Count positive and negative keywords
            pos_count = sum(content.count(word) for word in positive_keywords)
            neg_count = sum(content.count(word) for word in negative_keywords)
            sentiment_score += (pos_count - neg_count)

        # Normalize and clamp sentiment to [0, 10]
        if sentiment_score < 0:
            sentiment_score = 0
        elif sentiment_score > 10:
            sentiment_score = 10

        return sentiment_score

    except requests.RequestException as e:
        print(f"[Error] Unable to fetch public sentiment data for {coin_name}: {e}")
        return 0

# --- 7-STEP RECOMMENDATION FUNCTION ---
def prepare_shortlist():
    """
    Fetch data from CoinMarketCap and filter coins (excluding sentiment).
    Returns a shortlist of up to 20 coins that still need sentiment scoring.
    """
    data = fetch_market_data()
    if not data:
        return None, "Failed to fetch market data. Please try again later."

    # Basic Filters
    min_market_cap = 1_000_000
    max_market_cap = 200_000_000
    min_volume = 10_000
    reference_market_cap = 50_000_000

    filtered_coins = []
    for coin in data:
        name = coin.get("name", "")
        symbol = coin.get("symbol", "").upper()
        quote_data = coin.get("quote", {}).get("USD", {})

        current_price = quote_data.get("price", 0)
        market_cap = quote_data.get("market_cap", 0)
        volume_24h = quote_data.get("volume_24h", 0)
        change_24h = quote_data.get("percent_change_24h", 0) or 0
        change_7d = quote_data.get("percent_change_7d", 0) or 0
        cmc_rank = coin.get("cmc_rank", None)

        if current_price <= 0:
            continue
        if not (min_market_cap <= market_cap <= max_market_cap):
            continue
        if volume_24h < min_volume:
            continue

        # Potential return
        potential_return_raw = reference_market_cap / market_cap
        potential_return = round(potential_return_raw, 2)

        # Base score
        momentum = abs(change_24h) + abs(change_7d)
        liquidity_raw = (volume_24h / market_cap) * 10
        liquidity_score = min(liquidity_raw, 10) * 0.5
        base_score = (potential_return * 2) + (momentum * 0.1) + liquidity_score

        filtered_coins.append({
            "name": name,
            "rank": cmc_rank,
            "symbol": symbol,
            "current_price": current_price,
            "market_cap": market_cap,
            "volume_24h": volume_24h,
            "change_24h": change_24h,
            "change_7d": change_7d,
            "potential_return": potential_return,
            "base_score": base_score
        })

    if not filtered_coins:
        return None, "No coins passed the basic filters."

    # Sort by base_score, take top 20
    filtered_coins.sort(key=lambda c: c["base_score"], reverse=True)
    short_list = filtered_coins[:20]
    if not short_list:
        return None, "No coins in short list."

    return short_list, None

def finalize_coins_with_sentiment(short_list, dip_percent=5, target_profit_percent=50):
    """
    Loops over the short_list, fetches sentiment, computes final scores,
    then sorts and returns the top 3.
    """
    sentiment_weight = 1.2
    for coin in short_list:
        sentiment_score = fetch_public_sentiment(coin["name"])
        coin["sentiment_score"] = sentiment_score
        coin["final_score"] = coin["base_score"] + sentiment_score * sentiment_weight

    # Sort by final_score
    short_list.sort(key=lambda c: c["final_score"], reverse=True)
    top_coins = short_list[:3]
    if not top_coins:
        return "No speculative coins found after final scoring."

    # Build results
    results = []
    for coin in top_coins:
        entry_price = coin["current_price"] * (1 - dip_percent / 100)
        exit_price = entry_price * (1 + target_profit_percent / 100)
        stop_loss = entry_price * (1 - dip_percent / 100)
        rank_str = f"Rank: {coin['rank']}" if coin['rank'] else "Rank: N/A"
        block = (
            f"[b]{coin['name']} ({coin['symbol']})[/b]\n"
            f"{rank_str}\n"
            f"Current Price: [color=00FF00]${coin['current_price']:.6f}[/color]\n"
            f"Market Cap: [color=00FF00]{abbreviate_number(coin['market_cap'])}[/color]\n"
            f"24h Volume: [color=00FF00]{abbreviate_number(coin['volume_24h'])}[/color]\n"
            f"24h Change: [color=00FF00]{coin['change_24h']:.2f}%[/color]\n"
            f"7d Change: [color=00FF00]{coin['change_7d']:.2f}%[/color]\n"
            f"Potential Return: [color=FFD700]{coin['potential_return']}x[/color]\n"
            f"Sentiment Score: [color=FFD700]{coin['sentiment_score']}/10[/color]\n"
            f"Final Score: [color=FFD700]{coin['final_score']:.2f}[/color]\n\n"
            f"Suggested Entry: [color=00FF00]${entry_price:.6f}[/color]\n"
            f"Target Exit: [color=0000FF]${exit_price:.6f}[/color]\n"
            f"Stop Loss: [color=FF0000]${stop_loss:.6f}[/color]\n"
        )
        results.append(block)

    return "\n----------------\n".join(results)

class NoteCard(MDCard):
    """
    Dynamically resizes so large text doesn't overflow.
    """
    note_id = StringProperty()
    note_content = StringProperty()

    def __init__(self, note_id, content, edit_callback, delete_callback, **kwargs):
        super().__init__(**kwargs)
        self.note_id = note_id
        self.note_content = content
        self.edit_callback = edit_callback
        self.delete_callback = delete_callback

        # Let the card auto-size itself vertically
        self.size_hint_y = None
        self.height = dp(100)  # minimal initial height
        self.orientation = "vertical"
        self.padding = dp(10)
        self.spacing = dp(8)
        self.radius = [12, 12, 12, 12]
        self.md_bg_color = (1, 1, 1, 0.1)

        # Optional icon
        self.iicon = MDIconButton(
            icon="file-document",
            # replaced user_font_size with icon_size
            icon_size="20sp"
        )
        self.add_widget(self.iicon)

        # Label for the note content
        self.note_label = MDLabel(
            text=content,
            halign="left",
            theme_text_color="Primary",
            # These are key for auto-sizing:
            size_hint_y=None,
            text_size=(self.width - dp(20), None),  # We'll bind to width later
        )
        self.note_label.bind(
            texture_size=self.update_label_size
        )
        # We also bind the card's width so the label can reflow if the card is resized
        self.bind(width=self.on_card_width)

        self.add_widget(self.note_label)

        # Buttons row at bottom
        button_row = MDBoxLayout(
            orientation="horizontal",
            spacing="10dp",
            size_hint_y=None,
            height=dp(40)
        )
        edit_btn = MDRaisedButton(
            text="Edit",
            md_bg_color=(0.3, 0.5, 0.9, 1),
            on_release=lambda x: self.edit_callback(note_id, content)
        )
        delete_btn = MDRaisedButton(
            text="Delete",
            md_bg_color=(1, 0.2, 0.2, 1),
            on_release=lambda x: self.delete_callback(note_id)
        )
        button_row.add_widget(edit_btn)
        button_row.add_widget(delete_btn)
        self.add_widget(button_row)

    def on_card_width(self, *args):
        """
        Whenever the card's width changes, update label.text_size so it wraps properly.
        """
        # Subtract some padding to avoid clipping
        self.note_label.text_size = (self.width - dp(20), None)

    def update_label_size(self, *args):
        """
        Called whenever the label's texture_size changes,
        so we can update the label height + card height.
        """
        # The label's height should match its texture
        self.note_label.height = self.note_label.texture_size[1]

        # Extra vertical spacing for icon, buttons, and padding
        icon_height = dp(48)  # approx space for the icon row
        buttons_height = dp(48)  # approx space for the bottom button row
        padding_space = dp(20)  # top/bottom padding

        # The new total height
        self.height = self.note_label.height + icon_height + buttons_height + padding_space

class MyNotepadScreen(MDScreen):
    """
    A single screen that:
      - Has a fixed-height MDToolbar at top (title visible)
      - Provides a search bar that filters notes
      - Shows notes in a scrollable layout
      - Lets user create/edit notes via dialogs
    """

    def __init__(self, name, user_uid, **kwargs):
        super().__init__(name=name, **kwargs)
        self.user_uid = user_uid
        self.db = db

        # We'll store notes for filtering
        self.all_notes = []
        self.filtered_notes = []

        # ---------- MAIN LAYOUT (VERTICAL) ----------
        self.main_layout = MDBoxLayout(
            orientation="vertical",
            spacing="20dp",
            padding="20dp"
        )
        self.add_widget(self.main_layout)

        # 1) CREATE A FIXED-HEIGHT TOOLBAR
        self.toolbar = MDTopAppBar(
            title="My Notepad",
            elevation=10,
            md_bg_color=(0.2, 0.2, 0.4, 1),
            left_action_items=[["backspace", lambda x: MDApp.get_running_app().go_back()]]
        )

        # We'll add a text field for searching
        self.search_field = MDTextField(
            hint_text="Search notes...",
            mode="round",
            size_hint=(0.5, None),
            pos_hint={"center_x": 0.5},
            #height="40dp"
        )
        # We'll manually bind the text property

        self.search_field.bind(text=self.on_search_text)
        # Could add a right icon to the toolbar, but let's keep it simple
        self.toolbar.right_action_items = [["database-export", lambda x: MDApp.get_running_app().open_stock()]]
        #self.toolbar.add_widget(self.search_field)

        # Add the toolbar to the top of the layout
        self.main_layout.add_widget(self.toolbar)
        self.main_layout.add_widget(self.search_field)

        # 2) SCROLLABLE AREA FOR NOTES
        self.scroll_view = ScrollView()
        self.note_container = MDBoxLayout(
            orientation="vertical",
            spacing="10dp",
            padding="10dp",
            size_hint_y=None
        )
        self.note_container.bind(minimum_height=self.note_container.setter("height"))
        self.scroll_view.add_widget(self.note_container)
        self.main_layout.add_widget(self.scroll_view)
        self.fab1 = MDFloatingActionButton(
            icon="pencil",
            md_bg_color=(0.8, 0.7, 0, 1),
            pos_hint={"right": 0.95, "y": 0.22},
            on_release=self.on_nter
        )
        self.add_widget(self.fab1)
        # 3) FLOATING ACTION BUTTON FOR NEW NOTES
        self.fab = MDFloatingActionButton(
            icon="plus",
            md_bg_color=(0, 0.7, 0, 1),
            pos_hint={"right": 0.95, "y": 0.05},
            on_release=self.open_new_note_dialog
        )
        self.add_widget(self.fab)

    def on_nter(self, *args):
        """Refresh notes from Firebase whenever we enter the screen."""
        super().on_pre_enter(*args)
        self.fetch_notes()

    # ---------- SEARCH HANDLER ----------
    def on_search_text(self, instance, value):
        """
        Triggered whenever user types in the search field.
        We'll do a simple substring match (case-insensitive).
        """

        query = value.lower().strip()
        if not query:
            self.filtered_notes = list(self.all_notes)
        else:
            self.filtered_notes = [
                (nid, content)
                for (nid, content) in self.all_notes
                if query in content.lower()
            ]
        self.display_notes()

    # ---------- FETCH & DISPLAY ----------
    def fetch_notes(self):
        """Retrieve notes for this user from Firebase."""
        try:
            res = self.db.child("notes").child(self.user_uid).get()
            if not res.each():
                self.all_notes = []
            else:
                # Build list of (note_id, content)
                notes = []
                for item in res.each():
                    nid = item.key()
                    note_data = item.val()
                    txt = note_data.get("content", "")
                    notes.append((nid, txt))
                self.all_notes = notes

            # Initially, show all
            self.filtered_notes = list(self.all_notes)
            self.display_notes()

        except Exception as e:
            print(f"Error fetching notes: {e}")
            self.all_notes = []
            self.filtered_notes = []
            self.display_notes()

    def display_notes(self):
        self.note_container.clear_widgets()
        for (nid, content) in self.filtered_notes:
            note_card = NoteCard(
                note_id=nid,
                content=content,
                edit_callback=self.open_edit_note_dialog,
                delete_callback=self.delete_note,
                size_hint=(1, None)
            )
            self.note_container.add_widget(note_card)

    # ---------- CREATE NEW NOTE ----------
    def open_new_note_dialog(self, *args):
        self.dialog_text_field = MDTextField(
            multiline=True,
            hint_text="Write your note...",
            size_hint_y=None,
            height="200dp"
        )

        def close_dialog(_):
            if self.dialog:
                self.dialog.dismiss()

        def save_new_note(_):
            new_content = self.dialog_text_field.text.strip()
            if not new_content:
                print("[DEBUG] Empty note not saved.")
                return
            try:
                self.db.child("notes").child(self.user_uid).push({"content": new_content})
                print("[DEBUG] New note created.")
                self.fetch_notes()
            except Exception as e:
                print(f"[DEBUG] Error creating note: {e}")
            close_dialog(_)

        self.dialog = MDDialog(
            title="Create Note",
            type="custom",
            content_cls=self.dialog_text_field,
            buttons=[
                MDFlatButton(text="CANCEL", on_release=close_dialog),
                MDFlatButton(text="SAVE", on_release=save_new_note)
            ]
        )
        self.dialog.open()

    # ---------- EDIT NOTE ----------
    def open_edit_note_dialog(self, note_id, content):
        self.dialog_text_field = MDTextField(
            multiline=True,
            text=content,
            size_hint_y=None,
            height="200dp"
        )
        self.editing_note_id = note_id

        def close_dialog(_):
            if self.dialog:
                self.dialog.dismiss()

        def save_edited_note(_):
            updated_content = self.dialog_text_field.text.strip()
            if not updated_content:
                print("[DEBUG] Empty note not saved.")
                return
            try:
                self.db.child("notes").child(self.user_uid).child(self.editing_note_id).update({"content": updated_content})
                print("[DEBUG] Note updated.")
                self.fetch_notes()
            except Exception as e:
                print(f"[DEBUG] Error updating note: {e}")
            close_dialog(_)

        self.dialog = MDDialog(
            title="Edit Note",
            type="custom",
            content_cls=self.dialog_text_field,
            buttons=[
                MDFlatButton(text="CANCEL", on_release=close_dialog),
                MDFlatButton(text="SAVE", on_release=save_edited_note)
            ]
        )
        self.dialog.open()

    # ---------- DELETE NOTE ----------
    def delete_note(self, note_id):
        try:
            self.db.child("notes").child(self.user_uid).child(note_id).remove()
            print("[DEBUG] Note deleted.")
            self.fetch_notes()
        except Exception as e:
            print(f"[DEBUG] Error deleting note: {e}")

class Quran(FloatLayout):

    surah = ObjectProperty()
    ayah = ObjectProperty()
    first = StringProperty()
    second = StringProperty()
    third = StringProperty()
    fourth = StringProperty()
    fifth = StringProperty()

    def msg(self):
        download = Thread(target=self.good)
        download.start()

    def audi(self):
        download = Thread(target=self.good1)
        download.start()

    def good1(self):
        try:
            api = "http://api.alquran.cloud/v1/ayah/{}/ar.alafasy "
            result = requests.get(url=api.format(self.surah.text)).json()
            aud = result["data"]["audio"]
            if platform == 'android':
                from jnius import autoclass
                MediaPlayer = autoclass('android.media.MediaPlayer')
                self.player = MediaPlayer()
                self.player.setDataSource(aud)
                self.player.prepare()
                self.player.start()
        except:
            if platform == 'android':
                toast(text="Enter Ayah Number", gravity=80, y=200, x=0)
            else:
                toast(text="Enter Ayah Number")

    def backward(self):
        try:
            a = int(self.surah.text) - 1
            self.surah.text = str(a)
        except:
            if platform == 'android':
                toast(text="Enter Ayah Number", gravity=80, y=200, x=0)
            else:
                toast(text="Enter Ayah Number")

    def forward(self):
        try:
            a = int(self.surah.text) + 1
            self.surah.text = str(a)
        except:
            if platform == 'android':
                toast(text="Enter Ayah Number", gravity=80, y=200, x=0)
            else:
                toast(text="Enter Ayah Number")

    def good(self):
        try:
            api = "http://api.alquran.cloud/v1/ayah/{}/en.yusufali"
            result = requests.get(url=api.format(self.surah.text)).json()
            self.first = result['data']['surah']["englishName"]
            self.second = str(result['data']['surah']["number"])
            self.third = result['data']['surah']["englishNameTranslation"]
            self.fourth = str(result['data']['surah']["numberOfAyahs"])
            self.fifth = result['data']['surah']["revelationType"]
            self.ayah.text = result['data']['text']
        except:
            if platform == 'android':
                toast(text="Enter Ayah Number", gravity=80, y=200, x=0)
            else:
                toast(text="Enter Ayah Number")

    def gayab(self):
        self.ayah.text = "Enter Ayah number\n(For Example:1 to 6236)\n\nor\n\nSurah Number : Ayah Number\n(For Example:114:6)\n\nQuran recitation in audio by Mishary Rashid Alafasy"
        self.first = ""
        self.second = 0
        self.third = ""
        self.fourth = 0
        self.fifth = ""
        self.surah.text = ""

    def gaayab(self):
        self.surah.text = ""
    def copy_clipboard(self):
        """Copy the text of the recommendation results to the clipboard."""
        from kivy.core.clipboard import Clipboard
        Clipboard.copy(self.ayah.text)

class Openquran(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def on_enter(self):
        try:
            apii = "http://api.alquran.cloud/v1/ayah/1/ar.alafasy "
            result1 = requests.get(url=apii).json()
            aud1 = result1["data"]["audio"]
            if platform == 'android':
                from jnius import autoclass
                MediaPlayer = autoclass('android.media.MediaPlayer')
                self.player = MediaPlayer()
                self.player.setDataSource(aud1)
                self.player.prepare()
                self.player.start()
        except:
            toast(text="No Internet")

class TradingBotScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        layout = MDBoxLayout(orientation="vertical", padding=dp(10), spacing=dp(10))

        layout.add_widget(
            MDLabel(
                text="Trading Bot",
                halign="center",
                font_style="H4",
                size_hint_y=None,
                height=dp(40)
            )
        )

        # Text field for coin symbol input
        self.search_input = MDTextField(
            hint_text="Enter coin symbol",
            size_hint_x=0.8,
            pos_hint={"center_x": 0.5},
            mode="rectangle"
        )
        layout.add_widget(self.search_input)

        # Search button
        search_button = MDRaisedButton(
            text="Search Coin",
            #style="elevated",
            pos_hint={"center_x": 0.5},
            on_release=lambda x: MDApp.get_running_app().search_coin(self.search_input.text)
        )
        layout.add_widget(search_button)

        # Recommendation button
        fetch_button = MDRaisedButton(
            text="Get Trade Recommendations",
            #style="elevated",
            pos_hint={"center_x": 0.5},
            on_release=lambda x: MDApp.get_running_app().get_trade_recommendations()
        )
        layout.add_widget(fetch_button)
        dex_screener_button = MDRaisedButton(
            text="Get Latest Tokens (Dex Screener)",
            #style="elevated",
            pos_hint={"center_x": 0.5},
            on_release=lambda x: MDApp.get_running_app().fetch_and_display_top_tokens()
        )
        layout.add_widget(dex_screener_button)

        # Scrollable area for displaying recommendations
        scroll_view = ScrollView()
        self.recommendation_results = MDLabel(
            text="Click the button to get recommendations.",
            halign="left",
            valign="top",
            markup=True,
            size_hint_x=1,
            size_hint_y=None,
            text_size=(self.width - dp(40), None),
        )
        self.recommendation_results.bind(texture_size=self.recommendation_results.setter('size'))
        scroll_view.add_widget(self.recommendation_results)
        layout.add_widget(scroll_view)
        copy_button = MDRaisedButton(
            text="Copy Text",
            #style="elevated",
            pos_hint={"center_x": 0.5},
            on_release=lambda x: MDApp.get_running_app().copy_results_to_clipboard()
        )
        # Back button
        back_button = MDRaisedButton(
            text="Back",
            pos_hint={"center_x": 0.5},
            on_release=lambda x: MDApp.get_running_app().go_back()
        )
        # Back button
        ba_button = MDRoundFlatIconButton(
            text="VIP BOT",
            icon= "robot-excited",
            pos_hint={"center_x": 0.5},
            md_bg_color= "orange",
            theme_text_color= "Custom",
            text_color= (1,1,1,1),
            icon_color= (1,1,1,1),
            line_color = "orange",
            on_release=lambda x: MDApp.get_running_app().open_ab()
        )
        layout.add_widget(copy_button)
        layout.add_widget(ba_button)
        layout.add_widget(back_button)

        self.add_widget(layout)

class LoginScreen(Screen):
    pass

class SignupScreen(Screen):
    pass

class HomeScreen(Screen):
    pass
class Ab(Screen):
    coin_input = ObjectProperty()
    result_card_container = ObjectProperty()

class Abf(Screen):
    coin_input= ObjectProperty()
    progress_bar=ObjectProperty()
    header_card=ObjectProperty()
    header_label=ObjectProperty()
    reddit_card=ObjectProperty()
    reddit_label=ObjectProperty()
    indicators_card=ObjectProperty()
    indicators_label=ObjectProperty()
    fibonacci_card=ObjectProperty()
    fibonacci_label=ObjectProperty()
    fib_signal_label=ObjectProperty()
    volume_card=ObjectProperty()
    volume_label=ObjectProperty()
    votes_card=ObjectProperty()
    votes_label=ObjectProperty()
    reason_card=ObjectProperty()
    reason_label=ObjectProperty()

if platform == 'android':
    class ToastRunnable(PythonJavaClass):

        __javainterfaces__ = ['java/lang/Runnable']
        __javacontext__ = 'app'

        def __init__(self, toast):
            super(ToastRunnable, self).__init__()
            self.toast = toast

        @java_method('()V')
        def run(self):
            self.toast.show()

class CryptoCoachingApp(MDApp):
    dialog = None
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cached_coins = None  # Cache for the top 1000 coins list
        self.last_search_time = 0  # Initialize the last search time
        self.search_cooldown = 5  # Cooldown period in seconds
        self.request_interval = 1.5  # Minimum interval between requests in seconds
        self.last_request_time = 0
        self.db = None
        self.user_id = StringProperty()

    def build(self):
        self.sm = MDScreenManager()

        # Load KV files
        Builder.load_file("login_screen.kv")
        Builder.load_file("signup_screen.kv")
        Builder.load_file("home_screen.kv")
        Builder.load_file("one.kv")
        Builder.load_file("ab.kv")
        Builder.load_file("abf.kv")
        #Builder.load_file("posstock.kv")

        # Add screens to ScreenManager
        self.sm.add_widget(LoginScreen(name="login"))
        self.sm.add_widget(SignupScreen(name="signup"))
        self.sm.add_widget(HomeScreen(name="home"))
        self.sm.add_widget(Screen(name="quiz_results"))
        self.sm.add_widget(Openquran(name="one"))
        self.sm.add_widget(Ab(name="main_screen"))
        self.sm.add_widget(Abf(name="mfunc"))

        # If you already have a ScreenManager, add them:
        self.sm.add_widget(CompanyListScreen(name="company_list_screen"))
        self.sm.add_widget(ProductListScreen(name="product_list_screen"))
        self.sm.add_widget(TransactionScreen(name="transaction_screen"))

        nav_drawer = self.sm.get_screen("home").ids.nav_drawer
        nav_drawer.bind(state=self.update_backdrop_opacity)

        firebase = pyrebase.initialize_app(firebase_config)
        self.auth = firebase.auth()
        self.db = firebase.database()

        self.update_drawer_width()

        # Bind window size change to update drawer width
        Window.bind(on_resize=self.update_drawer_width)

        # Create theme menu
        self.theme_menu = None
        self.create_theme_menu()

        # Set Initial Screen
        self.sm.current = "login"
        Clock.schedule_once(self.load_courses, 1)

        # Bind the back button for Android
        Window.bind(on_keyboard=self.handle_back_button)
        Window.softinput_mode = "below_target"
        return self.sm

    def update_drawer_width(self, *args):
        """Update the navigation drawer width based on the current window size."""
        nav_drawer = self.sm.get_screen("home").ids.nav_drawer
        nav_drawer.width = Window.width * 0.7
    def create_theme_menu(self):
        """Create a dropdown menu for selecting themes."""
        themes = ["Default", "Red", "Pink", "Purple", "DeepPurple", "Indigo", "Blue", "LightBlue", "Cyan", "Teal", "Green", "LightGreen", "Lime", "Yellow", "Amber","Orange", "DeepOrange", "Brown", "Gray", "BlueGray"]
        menu_items = [
            {
                "text": theme,
                "viewclass": "OneLineListItem",
                "on_release": lambda x=theme: self.change_theme(x),
            }
            for theme in themes
        ]
        self.theme_menu = MDDropdownMenu(
            items=menu_items,
            width=dp(200),  # Explicit width
            position="center"  # Optional: Sets menu position
        )

    def open_theme_menu(self, instance):
        """Open the theme selection menu."""
        self.theme_menu.caller = instance
        self.theme_menu.open()

    def toggle_dark_mode(self, is_active):
        """Toggle between dark mode and light mode, and update UI colors."""
        if is_active:
            self.theme_cls.theme_style = "Dark"
            panel_color = (0.12, 0.12, 0.12, 1)  # Dark background
            text_color_active = (1, 1, 1, 1)  # White active icons
            text_color_normal = (0.7, 0.7, 0.7, 1)  # Gray inactive icons
            toast("Dark mode enabled")
        else:
            self.theme_cls.theme_style = "Light"
            panel_color = (0.98, 0.98, 0.98, 1)  # Light background
            text_color_active = (0, 0, 0, 1)  # Black active icons
            text_color_normal = (0.5, 0.5, 0.5, 1)  # Dark gray inactive icons
            toast("Light mode enabled")

        # Update colors dynamically
        home_screen = self.sm.get_screen("home")
        bottom_nav = home_screen.ids.bottom_nav
        bottom_nav.panel_color = panel_color
        bottom_nav.text_color_active = text_color_active
        bottom_nav.text_color_normal = text_color_normal

        self.save_theme_preferences()

    def change_theme(self, theme_name):
        """Change the primary color palette based on user selection."""
        theme_palettes = {
            "Default": "Blue",
            "LightBlue": "LightBlue",
            "Red": "Red",
            "Pink": "Pink",
            "Green": "Green",
            "LightGreen": "LightGreen",
            "Blue": "Blue",
            "Purple": "Purple",
            "DeepPurple": "DeepPurple",
            "Lime": "Lime",
            "Yellow": "Yellow",
            "Brown": "Brown",
            "Gray": "Gray",
            "BlueGray": "BlueGray",
            "Orange": "Orange",
            "Amber": "Amber",
            "DeepOrange": "DeepOrange",
            "Indigo": "Indigo",
            "Cyan": "Cyan",
            "Teal": "Teal",
        }
        selected_palette = theme_palettes.get(theme_name, "Blue")
        self.theme_cls.primary_palette = selected_palette
        self.save_theme_preferences()
        self.theme_menu.dismiss()  # Close the menu
        toast(f"Theme changed to {theme_name}")


    def save_theme_preferences(self):
        """Save the current theme style and primary palette to a file."""
        preferences = {
            "theme_style": self.theme_cls.theme_style,
            "primary_palette": self.theme_cls.primary_palette,
        }
        with open("theme_settings.json", "w") as file:
            json.dump(preferences, file)

    def load_theme_preferences(self):
        """Load theme preferences from a file, if it exists."""
        if os.path.exists("theme_settings.json"):
            with open("theme_settings.json", "r") as file:
                preferences = json.load(file)
                self.theme_cls.theme_style = preferences.get("theme_style", "Light")
                self.theme_cls.primary_palette = preferences.get("primary_palette", "Blue")

    def update_backdrop_opacity(self, instance, value):
        """Show or hide the overlay based on the drawer's state."""
        home_screen = self.sm.get_screen("home")
        overlay = home_screen.ids.overlay

        if value == "open":
            overlay.opacity = 0.5  # Show the overlay with 50% opacity
            overlay.md_bg_color = (0, 0, 0, 0.5)  # Ensure correct color
            print("Drawer opened: Showing dimming overlay")
        else:
            overlay.opacity = 0  # Hide the overlay
            print("Drawer closed: Hiding dimming overlay")

    def handle_back_button(self, window, key, *args):
        if key == 27:  # Android back button
            if self.sm.current == "signup":
                self.sm.current = "login"
            elif self.sm.current == "company_list_screen":
                self.sm.current = "home"
            elif self.sm.current == "one":
                self.sm.current = "home"
            elif self.sm.current == "main_screen":
                self.sm.current = "home"
            elif self.sm.current == "notepad_screen":
                self.sm.current = "home"
            elif self.sm.current == "trading_bot":
                self.sm.current = "home"
            return True
        return False

    def save_uid(self,uid):
        """Save the user UID locally in a file."""
        with open("user_uid.txt", "w") as f:
            f.write(uid)

    def load_uid(self):
        """Load the user UID from a local file, if it exists."""
        if os.path.exists("user_uid.txt"):
            with open("user_uid.txt", "r") as f:
                return f.read().strip()
        return None
    def save_token(self, token):
        """Save the token to a file for future sessions."""
        with open("user_token.txt", "w") as file:
            file.write(token)
        print("Token saved successfully")

    def load_token(self):
        """Load the token from the file if it exists."""
        if os.path.exists("user_token.txt"):
            with open("user_token.txt", "r") as file:
                return file.read()
        return None

    def save_refresh_token(self, refresh_token):
        """Save the refresh token to a file."""
        with open("refresh_token.txt", "w") as file:
            file.write(refresh_token)
        print("Refresh token saved successfully.")

    def is_internet_available(self):
        try:
            requests.get("http://google.com", timeout=2)
            return True
        except requests.ConnectionError:
            return False

    def login(self):
        """Regular email-password login."""
        screen = self.sm.get_screen("login")
        email = screen.ids.email.text
        password = screen.ids.password.text
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            id_token = user["idToken"]
            refresh_token = user["refreshToken"]
            uid = user["localId"]  # <-- This is the actual user UID
            print(f"User successfully logged in! UID: {uid}")
            # Save all tokens
            self.save_token(id_token)
            self.save_refresh_token(refresh_token)
            self.save_uid(uid)
            self.go_to_home()
        except Exception as e:
            dialog = MDDialog(title=get_error_message(e),
                              buttons=[MDFlatButton(text="OK", on_release=lambda x: dialog.dismiss())])
            dialog.open()

    def forgot_password(self):
        screen = self.sm.get_screen("login")
        email = screen.ids.email.text
        if not email:
            dialog = MDDialog(title="Please fill ur registered email id to get the password changing mail.",buttons=[MDFlatButton(text="OK",on_release=lambda x: dialog.dismiss())])
            dialog.open()
            return
        try:
            auth.send_password_reset_email(email)
            dialog = MDDialog(
                title="Password reset email sent",
                buttons=[
                    MDFlatButton(
                        text="OK",
                        on_release=lambda x: dialog.dismiss()
                    )
                ]
            )
            dialog.open()

        except Exception as e:
            dialog = MDDialog(
                title=get_error_message(e),
                buttons=[
                    MDFlatButton(
                        text="OK",
                        on_release=lambda x: dialog.dismiss()
                    )
                ]
            )
            dialog.open()

    def go_to_signup(self):
        self.sm.current = "signup"
        toast("Navigated to Signup Screen")


    def signup(self):
        screen = self.sm.get_screen("signup")
        email = screen.ids.email.text
        password = screen.ids.password.text
        try:
            user = auth.create_user_with_email_and_password(email, password)
            id_token = user["idToken"]
            refresh_token = user["refreshToken"]

            # Save both tokens
            self.save_token(id_token)
            self.save_refresh_token(refresh_token)
        except Exception as e:
            dialog = MDDialog(title=get_error_message(e),
                              buttons=[MDFlatButton(text="OK", on_release=lambda x: dialog.dismiss())])
            dialog.open()

    def go_to_login(self):
        """Navigate to the login screen."""
        print("Navigating to login screen...")
        self.sm.current = "login"

    def go_to_home(self):
        """Navigate to the home screen."""
        print("Navigating to home screen...")
        self.sm.current = "home"
        self.load_courses()
        #toast("Navigated to Home Screen")

    def on_start(self):
        """Called when the app starts."""
        if not self.is_internet_available():
            dialog = MDDialog(title="No internet connection. Please check your network.",
                              buttons=[MDFlatButton(text="OK", on_release=lambda x: dialog.dismiss())])
            dialog.open()

            return []

        token = self.load_token()
        if token and not is_token_expired(token):
            # If token is valid, attempt to load user profile
            self.load_user_profile()
            self.load_theme_preferences()
            self.go_to_home()
        else:
            # If no token or expired token, try refreshing first
            refreshed = self.refresh_token()
            if refreshed:
                # If refresh successful, load user profile
                self.load_user_profile()
                self.load_theme_preferences()
                self.go_to_home()
            else:
                # Otherwise go to login
                self.go_to_login()

    def update_courses(self, courses):
        """Update the courses label on the home screen."""
        screen = self.sm.get_screen("home")
        screen.ids.course_list.clear_widgets()

        for course in courses.split("\n"):
            screen.ids.course_list.add_widget(
                OneLineListItem(text=course)
            )
        print("Courses updated successfully")

    def load_user_profile(self):
        """Load the user's profile, refreshing the token if needed."""
        token = self.ensure_valid_token()
        if not token:
            return  # Token is invalid, already redirected to login

        try:
            user_info = auth.get_account_info(token)
            email = user_info["users"][0].get("email", "Unknown")
            photo_url = user_info["users"][0].get("photoUrl", "")
            local_id = user_info["users"][0]["localId"]# <-- This is the actual user UID
            print(f"User successfully logged in! UID: {local_id}")

            # Update UI with user information
            self.sm.get_screen("home").ids.profile_label.text = f"Logged in as: {email}"
            if photo_url:
                self.sm.get_screen("home").ids.profile_picture.source = photo_url
            else:
                self.sm.get_screen("home").ids.profile_picture.source = "default_profile.png"

            print("User profile loaded successfully.")
            self.go_to_home()

        except Exception as e:
            print(f"Error loading user profile: {e}")
            self.go_to_login()

    def load_refresh_token(self):
        """Load the refresh token from file if it exists."""
        if os.path.exists("refresh_token.txt"):
            with open("refresh_token.txt", "r") as file:
                return file.read()
        return None

    def refresh_token(self):
        """Refresh the Firebase ID token using the stored refresh token."""
        try:
            refresh_token = self.load_refresh_token()
            if not refresh_token:
                print("No refresh token available.")
                return None

            firebase_url = f"https://securetoken.googleapis.com/v1/token?key={firebase_config['apiKey']}"
            payload = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token
            }
            response = requests.post(firebase_url, data=payload, timeout=10)
            response_data = response.json()

            if 'id_token' in response_data:
                new_id_token = response_data['id_token']
                new_refresh_token = response_data['refresh_token']

                # Save the new tokens
                self.save_token(new_id_token)
                self.save_refresh_token(new_refresh_token)

                print("Token refreshed successfully.")
                return new_id_token
            else:
                print(f"Failed to refresh token: {response_data.get('error', 'Unknown error')}")
                return None
        except Exception as e:
            print(f"Error refreshing token: {e}")
            return None

    def ensure_valid_token(self):
        """
        Ensure the token is valid, refreshing it if necessary.
        Returns a valid token or None if refresh fails.
        """
        token = self.load_token()
        if token and is_token_expired(token):
            # Attempt refresh
            refreshed_token = self.refresh_token()
            if refreshed_token:
                return refreshed_token
            else:
                print("Failed to refresh token. Redirecting to login.")
                self.go_to_login()
                return None
        return token

    def _set_profile_picture(self, data):
        if data:
            self.sm.get_screen("home").ids.profile_picture.texture = data.texture
            toast("Profile picture loaded successfully")

    def google_sign_in(self):
        def auth_flow():
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "client_secrets.json",
                    scopes=["openid",
                            "https://www.googleapis.com/auth/userinfo.email",
                            "https://www.googleapis.com/auth/userinfo.profile"]
                )
                print("Flow initialized")

                credentials = flow.run_local_server(port=8080)
                print("Authentication completed")

                # Decode the ID token
                decoded_token = id_token.verify_oauth2_token(
                    credentials.id_token, Request()
                )
                email = decoded_token.get("email", "Unknown")
                photo_url = decoded_token.get("picture", "")  # user’s profile pic

                # Sign in with Google ID token via Firebase
                firebase_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithIdp?key={firebase_config['apiKey']}"
                payload = {
                    "postBody": f"id_token={credentials.id_token}&providerId=google.com",
                    "requestUri": "http://localhost",
                    "returnIdpCredential": True,
                    "returnSecureToken": True
                }
                response = requests.post(firebase_url, json=payload)
                response_data = response.json()

                if 'idToken' in response_data:
                    id_token_firebase = response_data['idToken']
                    refresh_token_firebase = response_data.get('refreshToken')

                    self.save_token(id_token_firebase)
                    uid = response_data["localId"]
                    self.local_id = response_data["localId"]
                    self.save_uid(uid)
                    if refresh_token_firebase:
                        self.save_refresh_token(refresh_token_firebase)

                    # UI update in main thread
                    def update_ui(dt):
                        # e.g., set labels, nav to home
                        toast("Google Sign-In successful")
                        self.update_ui(email, photo_url)

                    from kivy.clock import Clock
                    Clock.schedule_once(update_ui)

                    print("Google Sign-In successful")
                else:
                    error_message = response_data.get('error', {}).get('message', 'Unknown error')
                    raise Exception(f"Firebase sign-in failed: {error_message}")

            except Exception as e:
                error_message = str(e)
                print(f"Error during Google Sign-In: {error_message}")
                from kivy.clock import Clock
                Clock.schedule_once(lambda dt: toast(f"Google Sign-In failed: {error_message}"))

        Thread(target=auth_flow).start()

    def update_ui(self, email, photo_url):
        """Update the UI with the user's info on the main thread."""
        self.sm.get_screen("home").ids.profile_label.text = f"Logged in as: {email}"
        if photo_url:
            self.sm.get_screen("home").ids.profile_picture.source = photo_url
        else:
            self.sm.get_screen("home").ids.profile_picture.source = "default_profile.png"

        # Navigate to the home screen
        self.go_to_home()

    def copy_results_to_clipboard(self):
        """Copy the text of the recommendation results to the clipboard."""
        from kivy.core.clipboard import Clipboard
        screen = self.sm.get_screen("trading_bot")
        Clipboard.copy(screen.recommendation_results.text)
        toast("Results copied to clipboard!")

    def logout(self):
        """Log out the user, clear the tokens, and navigate to the login screen."""
        try:
            if os.path.exists("user_token.txt"):
                os.remove("user_token.txt")
            if os.path.exists("refresh_token.txt"):
                os.remove("refresh_token.txt")
            if os.path.exists("user_uid.txt"):
                os.remove("user_uid.txt")
            if os.path.exists("theme_settings.json"):
                os.remove("theme_settings.json")

            print("User token,theme,uid and refresh token cleared")

            self.sm.current = "login"
            toast("Logged out successfully")
        except Exception as e:
            print(f"Error during logout: {e}")
            toast("An error occurred while logging out")

    def load_courses(self, dt=None):
        """Load courses into the home screen."""
        course_list = self.sm.get_screen("home").ids.course_list
        course_list.clear_widgets()

        for course in courses:
            item = OneLineListItem(
                text=course['title'],
                theme_text_color="Custom",
                divider_color=(1, 1, 1, 1),
                text_color=(1, 1, 1, 1),
                on_release=partial(self.display_course, course)  # Use partial to bind the correct course
            )
            course_list.add_widget(item)

    def display_course(self, course, *args):
        """Display the selected course in a scrollable view with expandable topics and a quiz button."""
        screen = Screen(name="course_detail")

        layout = MDBoxLayout(orientation="vertical", padding=dp(10), spacing=dp(10), size_hint_y=None)
        layout.bind(minimum_height=layout.setter("height"))

        # Course title and description
        layout.add_widget(
            MDLabel(
                text=f"[b]{course['title']}[/b]",
                markup=True,
                halign="center",
                size_hint_y=None,
                font_style="H5"
            )
        )
        layout.add_widget(
            MDLabel(
                text=course['description'],
                halign="left",
                size_hint_y=None,
                text_size=(self.root.width - dp(40), None),
                valign="top"
            )
        )

        # Add topics as expansion panels
        for topic in course['topics']:
            content_box = MDBoxLayout(orientation="vertical", padding=dp(10), spacing=dp(5), size_hint_y=None)
            content_box.bind(minimum_height=content_box.setter("height"))

            content_label = MDLabel(
                text=topic['content'],
                halign="left",
                size_hint_y=None,
                text_size=(self.root.width - dp(40), None),
                valign="top"
            )
            content_label.bind(texture_size=content_label.setter('size'))
            content_box.add_widget(content_label)

            panel = MDExpansionPanel(
                icon="folder",
                content=content_box,
                panel_cls=MDExpansionPanelThreeLine(text=topic['title'])
            )
            layout.add_widget(panel)

        # Scrollable view for the course content
        scroll_view = ScrollView()
        scroll_view.add_widget(layout)

        # Fixed buttons at the bottom
        button_layout = MDBoxLayout(orientation="horizontal", size_hint_y=None, height=dp(50), spacing=dp(10))

        quiz_button = MDRaisedButton(
            text="Take Quiz",
            pos_hint={"center_x": 0.5},
            on_release=lambda x=course: self.show_course_quiz(course)
        )
        back_button = MDRaisedButton(
            text="Back",
            pos_hint={"center_x": 0.5},
            on_release=self.go_back_to_home
        )

        button_layout.add_widget(quiz_button)
        button_layout.add_widget(back_button)

        # Main layout combining scrollable content and fixed buttons
        main_layout = MDBoxLayout(orientation="vertical")
        main_layout.add_widget(scroll_view)
        main_layout.add_widget(button_layout)

        screen.add_widget(main_layout)
        self.sm.add_widget(screen)
        self.sm.current = "course_detail"

    def display_topic(self, topic):
        """Display the selected topic details, including quizzes and exercises."""
        screen = Screen(name="topic_detail")

        layout = MDCard(orientation="vertical", padding=dp(10), spacing=dp(10), size_hint_y=None)
        layout.bind(minimum_height=layout.setter("height"))

        layout.add_widget(Label(text=f"[b]{topic['title']}[/b]\n{topic['content']}",
                                markup=True, size_hint_y=None, text_size=(self.root.width - dp(40), None)))

        # Quiz Section
        quiz = topic.get('quiz')
        if quiz:
            quiz_button = MDRaisedButton(
                text="Take Quiz",
                on_release=lambda x=quiz: self.show_quiz(x)
            )
            layout.add_widget(quiz_button)

        # Exercise Section
        exercise = topic.get('exercise')
        if exercise:
            layout.add_widget(
                Label(text=f"Exercise: {exercise}", size_hint_y=None, text_size=(self.root.width - dp(40), None)))

        scroll_view = ScrollView()
        scroll_view.add_widget(layout)

        back_button = MDRaisedButton(
            text="Back", pos_hint={"center_x": 0.5}, on_release=self.go_back_to_course
        )

        main_layout = MDCard(orientation="vertical")
        main_layout.add_widget(scroll_view)
        main_layout.add_widget(back_button)

        screen.add_widget(main_layout)
        self.sm.add_widget(screen)
        self.sm.current = "topic_detail"

    def show_course_quiz(self, course):
        """Display the quiz for the selected course with a persistent Back button."""
        # Initialize quiz attributes
        self.current_question_index = 0
        self.user_answers = []
        self.correct_answers = 0

        if not self.sm.has_screen("quiz_screen"):
            screen = Screen(name="quiz_screen")
            self.sm.add_widget(screen)
        else:
            screen = self.sm.get_screen("quiz_screen")

        screen.clear_widgets()  # Clear previous content

        layout = MDBoxLayout(orientation="vertical", padding=dp(20), spacing=dp(20))

        # Title of the quiz
        title_label = MDLabel(
            text=f"Quiz: {course['title']}",
            halign="center",
            font_style="H5",
            size_hint_y=None
        )
        layout.add_widget(title_label)

        # Display the first question with number
        question_text = course['topics'][self.current_question_index]['quiz']['question']
        self.question_label = MDLabel(
            text=f"Question {self.current_question_index + 1}: {question_text}",
            halign="left",
            size_hint_y=None,
            text_size=(self.root.width - dp(40), None),
            color=get_color_from_hex("#1E88E5")  # Blue color for question text
        )
        layout.add_widget(self.question_label)

        # Input field for user answer
        self.answer_input = MDTextField(
            hint_text="Type your answer here",
            size_hint_x=0.8,
            pos_hint={"center_x": 0.5}
        )
        layout.add_widget(self.answer_input)

        # Submit button
        submit_button = MDRaisedButton(
            text="Submit Answer",
            pos_hint={"center_x": 0.5},
            on_release=lambda x: self.check_answer(course)
        )
        layout.add_widget(submit_button)

        # Back button
        back_button = MDRaisedButton(
            text="Back",
            pos_hint={"center_x": 0.5},
            on_release=self.go_back_to_course_detail
        )
        layout.add_widget(back_button)

        screen.add_widget(layout)
        self.sm.current = "quiz_screen"

    def check_answer(self, course):
        """Check the user's answer, provide feedback, and show the next question or results."""
        correct_answer = course['topics'][self.current_question_index]['quiz']['answer']
        user_answer = self.answer_input.text.strip()

        if user_answer.lower() == correct_answer.lower():
            self.correct_answers += 1
            toast("Correct!")
        else:
            toast(f"Wrong! The correct answer is: {correct_answer}")

        self.user_answers.append(user_answer)
        self.answer_input.text = ""  # Clear the input field

        self.current_question_index += 1

        if self.current_question_index < len(course['topics']):
            # Show the next question with updated number and color
            next_question = course['topics'][self.current_question_index]['quiz']['question']
            self.question_label.text = f"Question {self.current_question_index + 1}: {next_question}"
            self.question_label.color = get_color_from_hex("#1E88E5")
        else:
            # Show quiz results
            self.show_quiz_results(course)

    def show_quiz_results(self, course):
        """Update and display the final results of the quiz."""
        if not self.sm.has_screen("quiz_results"):
            screen = Screen(name="quiz_results")
            self.sm.add_widget(screen)
        else:
            screen = self.sm.get_screen("quiz_results")

        screen.clear_widgets()  # Clear any previous content

        layout = MDBoxLayout(orientation="vertical", padding=dp(20), spacing=dp(20))

        result_label = MDLabel(
            text=f"Quiz Completed!\nYou answered {self.correct_answers} out of {len(course['topics'])} questions correctly.",
            halign="center",
            font_style="H5",
            size_hint_y=None
        )
        layout.add_widget(result_label)

        back_button = MDRaisedButton(
            text="Back to Course",
            pos_hint={"center_x": 0.5},
            on_release=self.go_back_to_course_detail
        )
        layout.add_widget(back_button)

        screen.add_widget(layout)
        self.sm.current = "quiz_results"

    def go_back_to_course_detail(self, *args):
        """Navigate back to the course detail screen."""
        self.sm.current = "course_detail"

    def go_back_to_home(self, *args):
        self.sm.current = "home"
        self.sm.remove_widget(self.sm.get_screen("course_detail"))

    def go_back_to_course(self, *args):
        self.sm.current = "course_detail"
        self.sm.remove_widget(self.sm.get_screen("topic_detail"))


    def open_ab(self):
        """Navigate to the Trading Bot screen when button is clicked in the drawer."""
        if not self.sm.has_screen("main_screen"):
            trading_bot_scree = Ab(name="main_screen")
            self.sm.add_widget(trading_bot_scree)
        self.sm.current = "main_screen"
    def open_abf(self):
        """Navigate to the Trading Bot screen when button is clicked in the drawer."""
        if not self.sm.has_screen("mfunc"):
            trading_bot_scree = Abf(name="mfunc")
            self.sm.add_widget(trading_bot_scree)
        self.sm.current = "mfunc"
    def open_airdrops(self):
        """Navigate to the Trading Bot screen when button is clicked in the drawer."""
        if not self.sm.has_screen("notepad_screen"):
            uid = self.load_uid()
            trading_bot_scree = MyNotepadScreen(name="notepad_screen",user_uid=uid)
            self.sm.add_widget(trading_bot_scree)
        self.sm.current = "notepad_screen"
    def open_quran(self):
        """Navigate to the Trading Bot screen when button is clicked in the drawer."""
        if not self.sm.has_screen("one"):
            trading_bot_scree = Openquran(name="one")
            self.sm.add_widget(trading_bot_scree)
        self.sm.current = "one"
    def open_stock(self):
        """Navigate to the Trading Bot screen when button is clicked in the drawer."""
        if not self.sm.has_screen("company_list_screen"):

            trading_bot_scree = CompanyListScreen(name="company_list_screen")
            self.sm.add_widget(trading_bot_scree)
        self.sm.current = "company_list_screen"
    def open_trading_bot_screen(self):
        """Navigate to the Trading Bot screen when button is clicked in the drawer."""
        if not self.sm.has_screen("trading_bot"):
            trading_bot_screen = TradingBotScreen(name="trading_bot")
            self.sm.add_widget(trading_bot_screen)
        self.sm.current = "trading_bot"

    def get_trade_recommendations(self):
        """
        Called when the user presses the button (e.g., "Get Trade Recommendations").
        1) Prepare short list (fast).
        2) If error, show it immediately.
        3) If success, show progress dialog & start background thread for sentiment loop.
        """
        short_list, err = prepare_shortlist()
        screen = self.sm.get_screen("trading_bot")

        if err:
            screen.recommendation_results.text = err
            return

        # If short_list is good, let's show a progress dialog
        self._show_progress_dialog(len(short_list))

        # Then start background thread for sentiment
        def background_task():
            # We do partial progress updates for each coin in short_list
            total = len(short_list)
            for i, coin in enumerate(short_list, start=1):
                # 1) Fetch sentiment
                sentiment_score = fetch_public_sentiment(coin["name"])
                coin["sentiment_score"] = sentiment_score
                # 2) Compute final_score
                sentiment_weight = 1.2
                coin["final_score"] = coin["base_score"] + sentiment_score * sentiment_weight

                # 3) Update progress bar in main thread
                def update_progress(dt):
                    progress = (i / total) * 100
                    self.progress_bar.value = progress
                    self.progress_label.text = f"{progress:.0f}% Completed"
                Clock.schedule_once(update_progress, 0)

            # After loop, finalize (sort top 3, dismiss dialog, show results)
            def finalize(dt):
                # Sort
                short_list.sort(key=lambda c: c["final_score"], reverse=True)
                top_coins = short_list[:3]
                if not top_coins:
                    final_text = "No speculative coins found after final scoring."
                else:
                    # Build the final text
                    final_text = self._format_final_text(top_coins)

                # 1) Dismiss the dialog
                if self.progress_dialog:
                    self.progress_dialog.dismiss()

                # 2) Show results
                screen.recommendation_results.text = final_text

            Clock.schedule_once(finalize, 0)

        # Start the thread
        t = threading.Thread(target=background_task)
        t.daemon = True
        t.start()

    def _show_progress_dialog(self, list_size):
        """
        Create and open an MDDialog with an MDProgressBar and a label for percentage.
        """
        # Container for the progress bar & label
        content_box = MDBoxLayout(orientation="vertical", spacing="12dp", adaptive_height=True)

        # Create label & bar
        self.progress_label = MDLabel(text="0% Completed", halign="center")
        self.progress_bar = MDProgressBar(value=0, max=100)
        content_box.add_widget(self.progress_label)
        content_box.add_widget(self.progress_bar)

        self.progress_dialog = MDDialog(
            title="Wait for AHK Bot Magic",
            type="custom",
            content_cls=content_box,
            auto_dismiss=False
        )
        self.progress_dialog.open()

    def _format_final_text(self, top_coins):
        """
        Convert the top_coins data into the final results string.
        This is basically what finalize_coins_with_sentiment() was doing,
        but we'll do it inline for clarity.
        (If you prefer, keep using your 'finalize_coins_with_sentiment' function.)
        """
        dip_percent = 5
        target_profit_percent = 50
        results = []

        for coin in top_coins:
            entry_price = coin["current_price"] * (1 - dip_percent / 100)
            exit_price = entry_price * (1 + target_profit_percent / 100)
            stop_loss = entry_price * (1 - dip_percent / 100)
            rank_str = f"Rank: {coin['rank']}" if coin['rank'] else "Rank: N/A"
            block = (
                f"[b]{coin['name']} ({coin['symbol']})[/b]\n"
                f"{rank_str}\n"
                f"Current Price: [color=00FF00]${coin['current_price']:.6f}[/color]\n"
                f"Market Cap: [color=00FF00]{abbreviate_number(coin['market_cap'])}[/color]\n"
                f"24h Volume: [color=00FF00]{abbreviate_number(coin['volume_24h'])}[/color]\n"
                f"24h Change: [color=00FF00]{coin['change_24h']:.2f}%[/color]\n"
                f"7d Change: [color=00FF00]{coin['change_7d']:.2f}%[/color]\n"
                f"Potential Return: [color=FFD700]{coin['potential_return']}x[/color]\n"
                f"Sentiment Score: [color=FFD700]{coin['sentiment_score']}/10[/color]\n"
                f"Final Score: [color=FFD700]{coin['final_score']:.2f}[/color]\n\n"
                f"Suggested Entry: [color=00FF00]${entry_price:.6f}[/color]\n"
                f"Target Exit: [color=0000FF]${exit_price:.6f}[/color]\n"
                f"Stop Loss: [color=FF0000]${stop_loss:.6f}[/color]\n"
            )
            results.append(block)

        return "\n----------------\n".join(results)

    def go_back(self, *args):
        """Navigate back to the home screen from the Trading Bot screen."""
        self.sm.current = "home"

    def get_recommendation(self, total_score, trend, macd_signal, rsi):
        """
        Generate a recommendation based on total score, trend, MACD signal, and RSI.
        Returns:
            tuple: (recommendation message, trade type)
        """
        if trend == "Uptrend" or (trend == "Sideways" and np.any(macd_signal > 0) and rsi > 60):
            return "Take a Long Trade", "long"
        elif trend == "Downtrend" or (trend == "Sideways" and np.any(macd_signal < 0) and rsi < 40):
            return "Take a Short Trade", "short"
        else:
            return "It's not a good time for a trade", None

    def fetch_top_coins(self):
        """Fetch the top 1000 coins by market cap."""
        if not self.is_internet_available():
            toast("No internet connection. Please check your network.")
            return []
        try:
            url = "https://coingecko.p.rapidapi.com/coins/markets"
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 250,
                "page": 1
            }
            headers = {"X-RapidAPI-Key": ""}

            response = self.rate_limited_request(url, headers, params)
            response.raise_for_status()
            coins = response.json()

            # Check if data is returned correctly
            if not coins:
                print("No coins data found.")
            return coins

        except requests.RequestException as e:
            print(f"Error fetching top coins: {e}")
            return []

    def format_large_number(self, number):
        """Format large numbers into human-readable form with suffixes."""
        if number >= 1e12:
            return f"${number / 1e12:.2f}T"
        elif number >= 1e9:
            return f"${number / 1e9:.2f}B"
        elif number >= 1e6:
            return f"${number / 1e6:.2f}M"
        else:
            return f"${number:.2f}"

    def analyze_chart_pattern(self, prices):
        """
        Analyze price patterns using EMA, MACD, ADX, and RSI to predict trends.

        Args:
            prices (list): List of historical prices.

        Returns:
            str: Predicted trend ("Uptrend", "Downtrend", or "Sideways").
        """
        # Calculate technical indicators
        short_ema = self.calculate_ema(prices, window=12)
        long_ema = self.calculate_ema(prices, window=26)
        macd_line, signal_line = self.calculate_macd(prices)
        macd_signal = macd_line - signal_line  # Now both arrays have the same length
        adx = self.calculate_adx(prices)
        rsi_value = self.calculate_rsi(prices)

        # Determine trend based on multiple indicators
        if (short_ema[-1] > long_ema[-1] and macd_signal[-1] > 0) or (adx > 25 and rsi_value < 70):
            return "Uptrend"
        elif (short_ema[-1] < long_ema[-1] and macd_signal[-1] < 0) or (adx > 25 and rsi_value > 30):
            return "Downtrend"
        else:
            return "Sideways"

    def rate_limited_request(self, url, headers, params=None):
        """Perform a rate-limited request to avoid 429 errors."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.request_interval:
            time.sleep(self.request_interval - time_since_last_request)

        response = requests.get(url, headers=headers, params=params)
        self.last_request_time = time.time()  # Update the last request time
        response.raise_for_status()
        return response

    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index (RSI) for the given prices."""
        if len(prices) < window:
            return 50  # Neutral RSI if less data is available

        deltas = np.diff(prices)
        gains = deltas[deltas > 0].sum() / window
        losses = -deltas[deltas < 0].sum() / window

        if losses == 0:  # Avoid division by zero
            return 100

        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices, short_window=12, long_window=26, signal_window=9):
        """
        Calculate the MACD line and signal line.
        Args:
            prices (list): List of historical prices.
            short_window (int): Window for the short-term EMA.
            long_window (int): Window for the long-term EMA.
            signal_window (int): Window for the signal line EMA.
        Returns:
            tuple: (macd_line, signal_line) as NumPy arrays.
        """
        short_ema = self.calculate_ema(prices, window=short_window)
        long_ema = self.calculate_ema(prices, window=long_window)

        # Ensure both EMAs have the same length by trimming the start of the longer EMA
        min_length = min(len(short_ema), len(long_ema))
        short_ema = short_ema[-min_length:]
        long_ema = long_ema[-min_length:]

        macd_line = short_ema - long_ema
        signal_line = self.calculate_ema(macd_line, window=signal_window)

        # Trim macd_line to match the length of signal_line
        macd_line = macd_line[-len(signal_line):]

        return macd_line, signal_line

    def calculate_adx(self, prices, window=14):
        """
        Calculate the Average Directional Index (ADX) for trend strength.

        Args:
            prices (list): List of historical prices.
            window (int): Window size for ADX calculation.

        Returns:
            float: ADX value indicating trend strength.
        """
        prices = np.array(prices)  # Convert prices to a NumPy array

        # Calculate True Range (TR)
        tr = np.maximum.reduce([
            prices[1:] - prices[:-1],  # Price difference
            np.abs(prices[1:] - np.min(prices)),  # High - Previous Close
            np.abs(prices[1:] - np.max(prices))  # Low - Previous Close
        ])

        atr = np.convolve(tr, np.ones(window) / window, mode='valid')  # Average True Range

        # Calculate Directional Movement (DM)
        up_moves = np.maximum(prices[1:] - prices[:-1], 0)
        down_moves = np.maximum(prices[:-1] - prices[1:], 0)

        positive_dm = np.where(up_moves > down_moves, up_moves, 0)
        negative_dm = np.where(down_moves > up_moves, down_moves, 0)

        # Smooth positive and negative DM
        smoothed_positive_dm = np.convolve(positive_dm, np.ones(window) / window, mode='valid')
        smoothed_negative_dm = np.convolve(negative_dm, np.ones(window) / window, mode='valid')

        # Calculate Directional Indicator (DI)
        positive_di = 100 * (smoothed_positive_dm / atr)
        negative_di = 100 * (smoothed_negative_dm / atr)

        # Calculate DX and ADX
        dx = 100 * (np.abs(positive_di - negative_di) / (positive_di + negative_di))
        adx = np.mean(dx[-window:])  # Final ADX value as an average of the last window

        return adx

    def calculate_ema(self, prices, window):
        """Calculate the Exponential Moving Average (EMA) for the entire price series."""
        ema = [np.mean(prices[:window])]  # Start with the SMA of the first window
        multiplier = 2 / (window + 1)  # EMA multiplier formula

        for price in prices[window:]:
            ema.append((price - ema[-1]) * multiplier + ema[-1])

        return np.array(ema)

    def calculate_bollinger_bands(self, prices, window=20, num_std_dev=2):
        sma = np.mean(prices[-window:])
        std_dev = np.std(prices[-window:])
        upper_band = sma + (num_std_dev * std_dev)
        lower_band = sma - (num_std_dev * std_dev)
        return upper_band, lower_band



    def display_results(self, coin_name, symbol, rank, current_price, market_cap, volume,
                        price_change_percentage, trend, recommendation, entry_point_msg,
                        exit_point_msg, scores, trade_type):
        """
        Display the final results of the coin analysis in the trading bot screen.
        """
        screen = self.sm.get_screen("trading_bot")

        # Format large numbers for readability
        formatted_market_cap = self.format_large_number(market_cap)
        formatted_volume = self.format_large_number(volume)

        # Set entry/exit point colors based on recommendation
        entry_color = "00FF00" if trade_type == "long" else "FF0000"  # Green for long, red for short
        exit_color = "00FF00" if trade_type == "long" else "FF0000"

        # Prepare the result text
        result_text = f"""
    [b]Coin:[/b] {coin_name} ({symbol})
    [b]Rank:[/b] #{rank}
    [b]Current Price:[/b] ${current_price:.6f}
    [b]Market Cap:[/b] {formatted_market_cap}
    [b]Volume (24h):[/b] {formatted_volume}
    [b]Price Change (24h):[/b] {price_change_percentage:.2f}%
    [b]Trend Prediction:[/b] [color=00FF00]{trend}[/color]
    [b]Recommendation:[/b] [color=FFD700]{recommendation}[/color]

    [color={entry_color}]{entry_point_msg}[/color]
    [color={exit_color}]{exit_point_msg}[/color]

    [b]Scores:[/b]
    - Fundamental Analysis: {scores["fundamental"]:.1f}/10
    - Market Analysis: {scores["market"]:.1f}/10
    - Technical Analysis: {scores["technical"]:.1f}/10
    - Community Engagement: {scores["community"]:.1f}/10
    [b]Total Score:[/b] {scores["total"]}/10
    """

        # Display the result in the recommendation_results label
        screen.recommendation_results.text = result_text

    def calculate_scores(self, age_days, market_cap, volume, price_change_percentage_24h,
                      rank, prices, coin_name, community_data, developer_data):
        """
        Calculate scores for the coin based on various factors including fundamental, market, technical, and community aspects.

        Args:
            age_days (int): Age of the project in days.
            market_cap (float): Market capitalization in USD.
            volume (float): 24-hour trading volume in USD.
            price_change_percentage_24h (float): Percentage price change in the last 24 hours.
            prices (list): Historical prices for the coin.
            engagement (int): Community engagement score (e.g., social media activity).
            github_activity (int): Number of GitHub commits or contributions.
            rank (int): Rank of the coin by market cap.

        Returns:
            dict: A dictionary containing individual scores and the total score.
        """

        # Ensure valid inputs to prevent divide-by-zero errors
        market_cap = max(1, market_cap)
        volume = max(1, volume)




        # --- Market Analysis Score ---
        # Based on volume-to-market cap ratio and price change percentage
        volume_to_market_cap_ratio = volume / market_cap
        market_score = min(10, (volume_to_market_cap_ratio * 10) + (5 - abs(price_change_percentage_24h) / 2))

        # --- Technical Analysis Score ---
        # Use RSI, trend strength, and volatility to calculate the score
        rsi = self.calculate_rsi(prices)
        volatility = np.std(prices)
        trend = self.analyze_chart_pattern(prices)

        # RSI Contribution
        if 40 <= rsi <= 60:
            rsi_score = 10
        elif rsi < 30 or rsi > 70:
            rsi_score = 5  # Extreme RSI reduces the score
        else:
            rsi_score = 7

        # Trend Contribution
        if trend == "Uptrend" or trend == "Downtrend":
            trend_score = 8
        else:  # Sideways or unclear trends
            trend_score = 5

        # Volatility Contribution
        if volatility < 0.02:  # Very low volatility
            volatility_score = 5
        elif volatility > 0.1:  # High volatility
            volatility_score = 6
        else:  # Moderate volatility
            volatility_score = 8

        # Final technical score as a weighted average
        technical_score = round((rsi_score * 0.4) + (trend_score * 0.4) + (volatility_score * 0.2), 1)

        # --- Community Engagement Score ---
        twitter_followers = community_data.get("twitter_followers", 0)
        reddit_subscribers = community_data.get("reddit_subscribers", 0)

        max_followers = max(1e6, twitter_followers * 1.5)
        max_subscribers = max(1e5, reddit_subscribers * 1.5)

        community_score = min(10, (twitter_followers / max_followers) * 5 + (reddit_subscribers / max_subscribers) * 5)

        # --- Developer Score ---
        forks = developer_data.get("forks", 0)
        stars = developer_data.get("stars", 0)
        subscribers = developer_data.get("subscribers", 0)
        total_issues = developer_data.get("total_issues", 0)

        max_forks = 30000
        max_stars = 100000
        max_issues = 10000
        max_subscribers = 20000

        developer_score = min(10,
                              (forks / max_forks) * 2 + (stars / max_stars) * 3 + (total_issues / max_issues) * 2 + (
                                          subscribers / max_subscribers) * 3)

        # --- Sentiment Score ---
        price_change_weight = min(10, max(-10, price_change_percentage_24h / 5))
        social_activity_weight = min(5, (twitter_followers / 1e6) * 5)

        sentiment_score = 5 + price_change_weight + social_activity_weight
        sentiment_score = min(10, max(0, sentiment_score))  # Keep within [0, 10]

        # --- Fundamental Analysis Score ---
        # Based on project age, rank, and GitHub activity
        fundamental_score = min(10, (age_days / 365) * 2 + (10 - rank / 50) + (developer_score / 2))
        # --- Total Weighted Score ---
        total_score = round(
            (fundamental_score * 0.3) +
            (market_score * 0.2) +
            (technical_score * 0.3) +
            (community_score * 0.1) +
            (sentiment_score * 0.1),
            2)

        return {
            "fundamental": round(fundamental_score, 1),
            "market": round(market_score, 1),
            "technical": round(technical_score, 1),
            "community": round(community_score, 1),
            "sentiment": round(sentiment_score, 1),
            "total": total_score
        }

    def calculate_entry_exit_points(self, prices, current_price, trend, volatility, rsi):
        """
        Calculate realistic entry and exit points based on Bollinger Bands, support/resistance, trend, volatility, and RSI.

        Args:
            prices (list): List of historical prices.
            current_price (float): Current price of the coin.
            trend (str): Predicted trend ("Uptrend", "Downtrend", or "Sideways").
            volatility (float): Standard deviation of historical prices.
            rsi (float): Relative Strength Index (RSI) value.

        Returns:
            tuple: (entry_msg, exit_msg) containing suggested entry and exit points.
        """
        support = min(prices[-30:])  # Support level from last 30 days
        resistance = max(prices[-30:])  # Resistance level from last 30 days
        upper_band, lower_band = self.calculate_bollinger_bands(prices)  # Get Bollinger Bands

        if trend.lower() == "uptrend":
            if current_price < lower_band and rsi < 70:
                entry_point = lower_band  # Enter near the lower Bollinger Band
                exit_point = resistance  # Exit near resistance
            elif current_price > resistance:
                entry_point = current_price  # Enter at current price if it's breaking out
                exit_point = resistance * (1 + volatility / 100)  # Exit slightly above resistance
            else:
                entry_point = support  # Enter near support
                exit_point = upper_band  # Exit near the upper Bollinger Band

        elif trend.lower() == "downtrend":
            if current_price > upper_band and rsi > 30:
                entry_point = upper_band  # Short near the upper Bollinger Band
                exit_point = support  # Exit near support
            elif current_price < support:
                entry_point = current_price  # Short at current price if it's breaking down
                exit_point = support * (1 - volatility / 100)  # Exit slightly below support
            else:
                entry_point = resistance  # Short near resistance
                exit_point = lower_band  # Exit near the lower Bollinger Band

        elif trend.lower() == "sideways":
            entry_point = lower_band  # Enter near the lower Bollinger Band
            exit_point = upper_band  # Exit near the upper Bollinger Band

        else:
            entry_msg = "Trend is unclear. No trade recommended."
            exit_msg = "No exit point recommended."
            return entry_msg, exit_msg

        entry_msg = f"Suggested Entry Point: ${entry_point:.6f} (near {'support' if trend.lower() != 'sideways' else 'lower band'})"
        exit_msg = f"Suggested Exit Point: ${exit_point:.6f} (near {'resistance' if trend.lower() != 'sideways' else 'upper band'})"

        return entry_msg, exit_msg

    def search_coin(self, search_query):
        """Fetch and display detailed recommendation for a specific coin, including analysis, scoring, and entry/exit points."""

        if not search_query:
            toast("Please enter a valid coin symbol.")
            return

        screen = self.sm.get_screen("trading_bot")
        screen.recommendation_results.text = f"Searching for '{search_query.upper()}'..."

        def fetch_data(dt):
            query = search_query.strip().lower()
            top_coins = self.fetch_top_coins()

            # Find the exact match by symbol or name
            match = next(
                (coin for coin in top_coins if coin["symbol"].lower() == query or coin["name"].lower() == query),
                None
            )

            if not match:
                screen.recommendation_results.text = f"No data found for '{search_query}'. Please check the symbol or name."
                return

            coin_id = match["id"]

            try:
                # Fetch detailed market data
                response = requests.get(
                    f"https://coingecko.p.rapidapi.com/coins/{coin_id}",
                    headers=HEADERS
                )
                response.raise_for_status()
                coin_data = response.json()

                # Extract relevant data
                current_price = coin_data["market_data"]["current_price"]["usd"]
                market_cap = coin_data["market_data"]["market_cap"]["usd"]
                volume = coin_data["market_data"]["total_volume"]["usd"]
                price_change_percentage = coin_data["market_data"]["price_change_percentage_24h"]
                rank = coin_data["market_cap_rank"]
                coin_name = coin_data["name"]
                symbol = coin_data["symbol"].upper()

                # Calculate project age in days
                genesis_date = coin_data.get("genesis_date")
                if genesis_date:
                    genesis_date_obj = datetime.strptime(genesis_date, "%Y-%m-%d")
                    age_days = (datetime.now() - genesis_date_obj).days
                else:
                    age_days = 0

                # Fetch historical prices for technical analysis
                response = requests.get(
                    f"https://coingecko.p.rapidapi.com/coins/{coin_id}/market_chart",
                    params={"vs_currency": "usd", "days": 30},
                    headers=HEADERS
                )
                response.raise_for_status()
                prices = [point[1] for point in response.json()["prices"]]

                if prices and len(prices) > 0:
                    trend = self.analyze_chart_pattern(prices)  # Analyze chart pattern
                    rsi_value = self.calculate_rsi(prices)  # Calculate RSI
                    volatility = np.std(prices)  # Calculate volatility
                    macd_line, signal_line = self.calculate_macd(prices)
                else:
                    trend = "No Data"
                    rsi_value = 50  # Neutral RSI when no data is available
                    volatility = 0
                    macd_line, signal_line = 0, 0



                # Static or calculated engagement and activity scores
                #engagement = coin_data.get("community_score", 50)
                #github_activity = coin_data.get("developer_score", 20)
                # Fetch community and developer data
                community_data = coin_data.get("community_data", {})
                developer_data = coin_data.get("developer_data", {})

                # Calculate scores
                scores = self.calculate_scores(
                    age_days=age_days,
                    market_cap=market_cap,
                    volume=volume,
                    price_change_percentage_24h=price_change_percentage,
                    #engagement=engagement,
                    #github_activity=github_activity,
                    rank=rank,
                    prices=prices,
                    coin_name=coin_name,
                    community_data=community_data,
                    developer_data=developer_data
                )

                # Get recommendation based on scores and trend
                # Get recommendation based on scores and trend
                # Get recommendation based on scores, trend, RSI, and MACD
                recommendation, trade_type = self.get_recommendation(
                    total_score=scores["total"],
                    trend=trend,
                    rsi=rsi_value,
                    macd_signal=macd_line - signal_line
                )
                # Always calculate entry and exit points if trade_type is valid
                entry_msg, exit_msg = self.calculate_entry_exit_points(
                    prices=prices,
                    current_price=current_price,
                    trend=trend,
                    volatility=volatility,
                    rsi=rsi_value
                ) if trade_type else (
                    "It's not a good time to enter the market.",
                    "It's not a good time to exit the market."
                )

                # Display results
                self.display_results(
                    coin_name=coin_name,
                    symbol=symbol,
                    current_price=current_price,
                    market_cap=market_cap,
                    volume=volume,
                    price_change_percentage=price_change_percentage,
                    trend=trend,
                    recommendation=recommendation,
                    trade_type=trade_type,  # Pass trade_type here
                    entry_point_msg=entry_msg,
                    exit_point_msg=exit_msg,
                    scores=scores,
                    rank=rank
                )

            except requests.RequestException as e:
                screen.recommendation_results.text = f"Error fetching detailed data: {e}"
                print(f"HTTP error while fetching detailed data: {e}")

        Clock.schedule_once(fetch_data, 0)

        # ------------------- KivyMD App -------------------



    def fetch_and_display_top_tokens(self):
        """
        Display a progress dialog, run data fetch in a background thread.
        """
        # 1) Build progress dialog layout
        progress_layout = BoxLayout(orientation="vertical", size_hint=(1, None))

        progress_bar = MDProgressBar(value=0, max=100, size_hint=(1, None), height="20dp")
        progress_label = MDLabel(
            text="0%",

            halign="center",
            theme_text_color="Custom",
            text_color=[0, 0, 0, 1]
        )
        title_label = MDLabel(
            text="Fetching Solana Tokens from DEX...",
            halign="center",
            theme_text_color="Custom",
            text_color=[0, 0, 0, 1],
            font_style="H6",

        )
        progress_layout.add_widget(title_label)
        progress_layout.add_widget(progress_bar)
        progress_layout.add_widget(progress_label)

        self.progress_dialog = MDDialog(
            title="",
            type="custom",
            content_cls=progress_layout,
            md_bg_color=[1, 1, 1, 1],
            size_hint=(0.8, None),
            #height=dp(150),

            auto_dismiss=False
        )
        self.progress_dialog.open()

        def update_progress(current, total):
            percent = int((current / total) * 100)
            progress_bar.value = percent
            progress_label.text = f"{percent}%"

        def fetch_data():
            try:
                response = requests.get(
                    "https://api.dexscreener.com/token-boosts/latest/v1",
                    headers={},
                )
                response.raise_for_status()
                tokens = response.json()

                total_tokens = len(tokens)
                token_data = []
                unique_symbols = set()  # To track unique tokens by symbol

                for index, token in enumerate(tokens):
                    token_address = token.get("tokenAddress", "")
                    if not token_address:
                        continue

                    pair_response = requests.get(f"https://api.dexscreener.com/latest/dex/tokens/{token_address}")
                    pair_response.raise_for_status()
                    pair_data = pair_response.json()

                    if not pair_data.get("pairs"):
                        continue

                    pair_info = pair_data["pairs"][0]
                    base_token = pair_info["baseToken"]
                    symbol = base_token.get("symbol", "Unknown")

                    # Skip if this symbol has already been added
                    if symbol in unique_symbols:
                        continue

                    unique_symbols.add(symbol)  # Add symbol to the set

                    pair_info = pair_data["pairs"][0]
                    base_token = pair_info["baseToken"]
                    price_usd = pair_info.get("priceUsd", "N/A")
                    liquidity_usd = pair_info.get("liquidity", {}).get("usd", 0)
                    volume_usd = pair_info.get("volume", {}).get("h24", 0)
                    market_cap = pair_info.get("marketCap", 0)

                    from datetime import datetime, timezone
                    try:
                        pair_created_at = datetime.fromtimestamp(pair_info["pairCreatedAt"] // 1000,
                                                                 timezone.utc).strftime(
                            "%Y-%m-%d %H:%M:%S")
                    except KeyError:
                        pair_created_at = "Unknown"

                    image_url = pair_info.get("info", {}).get("imageUrl", "cph.png")
                    websites = pair_info.get("info", {}).get("websites", [])
                    twitter = next(
                        (s["url"] for s in pair_info.get("info", {}).get("socials", []) if s["type"] == "twitter"),
                        "No Twitter link available")
                    telegram = next(
                        (s["url"] for s in pair_info.get("info", {}).get("socials", []) if s["type"] == "telegram"),
                        "No Telegram link available")
                    website_url = next((site["url"] for site in websites if site["label"].lower() == "website"),
                                       "No website available")

                    token_data.append({
                        "name": base_token.get("name", "Unknown"),
                        "symbol": base_token.get("symbol", "Unknown"),
                        "chain": pair_info.get("chainId", "Unknown"),
                        "price": price_usd,
                        "liquidity": format_large_number(liquidity_usd),
                        "volume": format_large_number(volume_usd),
                        "market_cap": format_large_number(market_cap),
                        "price_change_24h": pair_info["priceChange"].get("h24", "N/A"),
                        "pair_created_at": pair_created_at,
                        "url": pair_info.get("url", "No URL available"),
                        "image_url": image_url,
                        "website": website_url,
                        "twitter": twitter,
                        "telegram": telegram,
                    })

                    # Update progress bar
                    mainthread(update_progress)(index + 1, total_tokens)
                    time.sleep(0.1)  # Simulate delay for progress visualization

                # Sort by market cap and volume
                sorted_tokens = sorted(
                    token_data,
                    key=lambda t: (
                        float(t["market_cap"].replace('k', '').replace('m', '').replace('b', '').replace('t', '')),
                        float(t["volume"].replace('k', '').replace('m', '').replace('b', '').replace('t', ''))
                    ),
                    reverse=True
                )[:5]

                @mainthread
                def display_results():
                    """Display the sorted tokens in a dialog."""
                    content = BoxLayout(orientation="vertical", size_hint_y=None)
                    content.bind(minimum_height=content.setter('height'))  # Bind height to fit content

                    scroll_view = ScrollView(size_hint=(1, None), height="500dp")
                    grid = MDGridLayout(cols=1, spacing=10, adaptive_height=True)

                    for token in sorted_tokens:
                        grid.add_widget(TokenCard(token))

                    scroll_view.add_widget(grid)

                    dismiss_button = MDRaisedButton(
                        text="Dismiss",
                        pos_hint={"center_x": 0.5},
                        on_release=lambda x: dialog.dismiss()
                    )
                    content.add_widget(scroll_view)
                    content.add_widget(dismiss_button)

                    dialog = MDDialog(
                        title="Top 5 Tokens",
                        type="custom",
                        content_cls=content,
                        size_hint=(0.9, 0.9),
                        auto_dismiss=True,
                    )
                    dialog.dismiss()
                    dialog.open()

                display_results()

            except requests.RequestException as e:
                print(f"HTTP error while fetching top tokens: {e}")
            except KeyError as e:
                print(f"Data parsing error: missing key {e}")
            finally:
                mainthread(self.progress_dialog.dismiss)()

        # Run the data fetching in a separate thread
        Thread(target=fetch_data).start()
    def on_analyze(self):
        """
        Called when user clicks 'Analyze'.
        We'll create + open a progress dialog, then run the background thread.
        """
        try:
            from kivy.factory import Factory
            content = Factory.ProgressDialogContent()
            self.progress_dialog = MDDialog(
                title="Analyzing...",
                type="custom",
                content_cls=content,
                auto_dismiss=False
            )
            self.progress_dialog.open()

            self.update_progress(0)
            # Start background thread
            thread = threading.Thread(target=self.background_analysis)
            thread.start()
        except Exception as e:
            self.show_error(f"Unexpected error: {str(e)}")


    def background_analysis(self):
        """
        The main workflow, run in a separate thread so we don't block the UI.
        We'll do each step, updating the progress bar and building final results.
        """
        try:
            # Step 1: Search coin
            self.step("Searching coin...", 20, self.search_coin_step)

            # Step 2: Fetch metadata
            self.step("Fetching metadata...", 40, self.metadata_step)

            # Step 3: Fetch OHLC & volume
            self.step("Fetching OHLC & volume...", 60, self.ohlc_step)

            # Step 4: TA + Reddit
            self.step("Applying indicators + sentiment...", 80, self.ta_sentiment_step)

            # Step 5: Future trade
            self.step("Generating final report...", 100, self.future_trade_step)

            # Done
            self.dismiss_dialog()

        except Exception as e:
            self.display_final_report()
            self.update_progress(0)
            self.dismiss_dialog()

    def step(self, debug_text, new_percent, step_func):
        """
        A utility to run one step, handle any exception, then update progress.
        We'll do short sleeps so the user sees the bar move in real time.
        """
        print(debug_text)
        step_func()
        time.sleep(0.5)  # short pause
        self.update_progress(new_percent)

    # ======================= STEPS =======================

    def search_coin_step(self):
        self.screen = self.sm.get_screen("main_screen")
        # screen.recommendation_results.text = f"Searching for '{search_query.upper()}'..."
        coin_name = self.screen.coin_input.text.strip()
        if not coin_name:
            raise ValueError("No coin name provided.")
        cid = self.search_coin_id(coin_name)
        if not cid:
            raise ValueError(f"No coin found for '{coin_name}'.")
        self.coin_id = cid

    def metadata_step(self):
        meta = self.fetch_coin_metadata(self.coin_id)
        if not meta:
            raise ValueError("Could not fetch coin metadata.")
        self.meta = meta

    def ohlc_step(self):
        data = self.fetch_ohlc_and_volume(self.coin_id, DEFAULT_DAYS)
        df_ohlc = self.prepare_ta_dataframe(data["ohlc"], data["market_chart"])
        if df_ohlc.empty:
            raise ValueError("No OHLC data returned.")
        self.df_ohlc = df_ohlc

    def ta_sentiment_step(self):
        df_ta = self.apply_indicators(self.df_ohlc)
        self.df_ta = df_ta

        coin_name = self.meta.get("name","")
        self.sentiment_val = self.reddit_sentiment(coin_name, limit=50)

    def future_trade_step(self):
        meta_price = self.meta.get("current_price", 0)
        self.future_trade = self.generate_future_trade(self.df_ta, meta_price=meta_price)

        # Build final
        final_text = self.format_final_report()
        self.set_result_label(final_text)

    # ======================= HELPER LOGIC =======================

    def search_coin_id(self, query):
        """
        If you want to pass a key, uncomment lines below.
        """
        url = f"{BASE_URL}/search"
        params = {"query": query.lower()}
        if COINGECKO_API_KEY:
            params["x_cg_demo_api_key"] = COINGECKO_API_KEY
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            return None
        data = resp.json()
        coins = data.get("coins", [])
        if not coins:
            return None
        coins.sort(key=lambda c: c.get("score",0), reverse=True)
        # attempt exact matches
        for c in coins:
            if c["id"].lower() == query.lower():
                return c["id"]
        for c in coins:
            if c["name"].lower() == query.lower():
                return c["id"]
        for c in coins:
            if c["symbol"].lower() == query.lower():
                return c["id"]
        return coins[0]["id"]

    def fetch_coin_metadata(self, coin_id):
        url = f"{BASE_URL}/coins/markets"
        params = {
            "vs_currency": "usd",
            "ids": coin_id
        }
        if COINGECKO_API_KEY:
            params["x_cg_demo_api_key"] = COINGECKO_API_KEY
        r = requests.get(url, params=params)
        if r.status_code != 200:
            return {}
        d = r.json()
        if not d:
            return {}
        return d[0]

    def fetch_ohlc_and_volume(self, coin_id, days=180):
        # OHLC
        url_ohlc = f"{BASE_URL}/coins/{coin_id}/ohlc"
        params1 = {"vs_currency":"usd","days":days}
        if COINGECKO_API_KEY:
             params1["x_cg_demo_api_key"] = COINGECKO_API_KEY
        r1 = requests.get(url_ohlc, params=params1)
        if r1.status_code != 200:
            raise Exception(f"OHLC error: {r1.text}")
        ohlc_data = r1.json()

        # Market chart
        url_mc = f"{BASE_URL}/coins/{coin_id}/market_chart"
        params2 = {"vs_currency":"usd","days":days,"interval":"daily"}
        # if COINGECKO_API_KEY:
        #     params2["x_cg_demo_api_key"] = COINGECKO_API_KEY
        r2 = requests.get(url_mc, params=params2)
        if r2.status_code != 200:
            raise Exception(f"Market chart error: {r2.text}")
        mc_data = r2.json()

        return {"ohlc": ohlc_data, "market_chart": mc_data}

    def prepare_ta_dataframe(self, ohlc_list, market_chart_data):
        if not ohlc_list:
            return pd.DataFrame()
        df = pd.DataFrame(ohlc_list, columns=["timestamp","open","high","low","close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        volumes = market_chart_data.get("total_volumes", [])
        if volumes:
            vdf = pd.DataFrame(volumes, columns=["ts","volume"])
            vdf["ts"] = pd.to_datetime(vdf["ts"], unit="ms")
            df = pd.merge_asof(
                df.sort_values("timestamp"),
                vdf.sort_values("ts"),
                left_on="timestamp",
                right_on="ts",
                direction="nearest"
            )
            df.rename(columns={"volume":"real_volume"}, inplace=True)
            df.drop(columns=["ts"], inplace=True)
        else:
            df["real_volume"] = 0
        return df

    def apply_indicators(self, df):
        df = df.copy()
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        df["rsi"] = ta.rsi(df["close"], length=14)

        macd_res = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd_res is None or macd_res.empty:
            df["macd"] = np.nan
            df["macd_signal"] = np.nan
            df["macd_hist"] = np.nan
        else:
            df["macd"] = macd_res["MACD_12_26_9"]
            df["macd_signal"] = macd_res["MACDs_12_26_9"]
            df["macd_hist"] = macd_res["MACDh_12_26_9"]

        bb_res = ta.bbands(df["close"], length=20, std=2.0)
        if bb_res is None or bb_res.empty:
            df["bb_lower"] = np.nan
            df["bb_mid"]   = np.nan
            df["bb_upper"] = np.nan
        else:
            df["bb_lower"] = bb_res["BBL_20_2.0"]
            df["bb_mid"]   = bb_res["BBM_20_2.0"]
            df["bb_upper"] = bb_res["BBU_20_2.0"]

        df.reset_index(inplace=True)
        return df

    def reddit_sentiment(self, coin_name, limit=100):
        """
        A simple average sentiment score from multiple subreddits using Afinn.
        If no credentials are provided, returns 0.0.
        """
        if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET):
            return 0.0
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        analyzer = Afinn()

        total_scores = []
        for s in SUBREDDITS:
            sub = reddit.subreddit(s)
            results = sub.search(coin_name, limit=limit)
            local = []
            for post in results:
                text = (post.title or "") + " " + (post.selftext or "")
                # Afinn returns a single score directly.
                score = analyzer.score(text)
                local.append(score)
            if local:
                total_scores.append(sum(local) / len(local))

        if not total_scores:
            return 0.0
        return sum(total_scores) / len(total_scores)

    def analyze_smc(self, df):
        if df.empty:
            return {"status":"No data","score":5}
        last_close = df.iloc[-1]["close"]
        mean_close = df["close"].mean()
        if last_close > mean_close:
            return {"status":"Accumulation","score":7}
        else:
            return {"status":"Distribution","score":4}

    def generate_future_trade(self, df, meta_price=0):
        if df.empty:
            return {"trade_type":"NO_TRADE","reason":"No data","entries":[],"exits":[]}

        last = df.iloc[-1]
        close = meta_price if (meta_price and meta_price>0) else last["close"]
        rsi = last["rsi"]
        macd_hist = last["macd_hist"]
        bb_lower = last["bb_lower"]
        bb_mid   = last["bb_mid"]
        bb_upper = last["bb_upper"]

        smc = self.analyze_smc(df)
        smc_status = smc["status"]

        bullish_score = 0
        bearish_score = 0

        # RSI weighting
        if pd.notna(rsi):
            if rsi < 40: bullish_score += 2
            elif rsi > 60: bearish_score += 2

        # MACD weighting
        if pd.notna(macd_hist):
            if macd_hist > 0: bullish_score += 2
            else: bearish_score += 2

        # SMC
        if smc_status == "Accumulation":
            bullish_score += 1
        else:
            bearish_score += 1

        if bullish_score > bearish_score:
            # LONG
            reason = f"RSI={rsi:.2f}, MACD={macd_hist}, SMC={smc_status} => bullish tilt"
            entry1 = close
            entry2 = np.nan
            if pd.notna(bb_lower) and bb_lower < close:
                entry2 = (close + bb_lower)/2

            if pd.notna(bb_mid):
                exit1 = max(bb_mid, close*1.05)
            else:
                exit1 = close*1.05

            if pd.notna(bb_upper):
                exit2 = max(bb_upper, exit1*1.05)
            else:
                exit2 = exit1*1.1

            return {
                "trade_type":"LONG",
                "reason":reason,
                "entries":[
                    {"label":"Entry #1", "price":entry1, "explanation":"Current price entry"},
                    {"label":"Entry #2", "price":entry2, "explanation":"If price dips toward lower band"}
                ],
                "exits":[
                    {"label":"Take Profit #1", "price":exit1, "explanation":"Take Profit"},
                    {"label":"Take Profit #2", "price":exit2, "explanation":"Bollinger Bands"}
                ]
            }
        elif bearish_score > bullish_score:
            # SHORT
            reason = f"RSI={rsi:.2f}, MACD={macd_hist}, SMC={smc_status} => bearish tilt"
            entry1 = close
            entry2 = np.nan
            if pd.notna(bb_upper) and bb_upper > close:
                entry2 = (close + bb_upper)/2

            if pd.notna(bb_mid):
                exit1 = min(bb_mid, close*0.95)
            else:
                exit1 = close*0.95

            if pd.notna(bb_lower):
                exit2 = min(bb_lower, exit1*0.95)
            else:
                exit2 = exit1*0.9

            return {
                "trade_type":"SHORT",
                "reason": reason,
                "entries":[
                    {"label":"Entry #1", "price":entry1, "explanation":"Short at current price"},
                    {"label":"Entry #2", "price":entry2, "explanation":"If price pops up"}
                ],
                "exits":[
                    {"label":"Take Profit #1", "price":exit1, "explanation":"Take Profit"},
                    {"label":"Take Profit #2", "price":exit2, "explanation":"Bollinger Bands"}
                ]
            }
        else:
            return {
                "trade_type":"NO_TRADE",
                "reason":"Neutral or conflicting signals",
                "entries":[],
                "exits":[]
            }

    def format_final_report(self):
        """
        Build final multi-line string with coin info, sentiment, RSI, MACD, future trade.
        We'll do dynamic price formatting so small prices (like SHIB) don't become $0.00
        """
        lines = []
        # metadata
        name = self.meta.get("name","")
        symbol = self.meta.get("symbol","").upper()
        cp = self.meta.get("current_price", 0)
        mc = self.meta.get("market_cap", 0)
        rk = self.meta.get("market_cap_rank", None)

        lines.append(f"Coin Name          :    {name}")
        lines.append(f"Symbol                :    {symbol}")
        lines.append(f"Rank                    :    {rk}")
        lines.append(f"Current Price      :    ${self.fmt_price(cp)}")
        lines.append(f"Market Cap         :    ${format_market_cap(mc)}")  # big number => 2 decimals is fine
        lines.append(f"Sentiment           :   {getattr(self, 'sentiment_val', 0.0):.3f}\n")

        # Show RSI + MACD
        if hasattr(self, 'df_ta') and not self.df_ta.empty:
            last = self.df_ta.iloc[-1]
            rsi_val = last.get("rsi", float('nan'))
            macd_val= last.get("macd_hist", float('nan'))
            lines.append(f"RSI                       :    {rsi_val:.2f}")
            lines.append(f"MACD Hist          :    {macd_val:.2f}\n")

        # future trade
        ft = getattr(self, 'future_trade', {})
        ttype = ft.get("trade_type","NO_TRADE")
        lines.append("--- FUTURE TRADE IDEA ---")
        if ttype == "NO_TRADE":
            lines.append("No strong future trade recommended.")
            lines.append(f"Reason: {ft.get('reason','N/A')}")
        else:
            lines.append(f"Trade Type: {ttype}")
            lines.append(f"Reason    : {ft.get('reason','N/A')}")

            lines.append("\nMultiple Entry Points:")
            for e in ft["entries"]:
                px = e["price"]
                if pd.notna(px):
                    lines.append(f"  {e['label']}: ${self.fmt_price(px)} ({e['explanation']})")

            lines.append("\nMultiple Exit Points:")
            for x in ft["exits"]:
                px = x["price"]
                if pd.notna(px):
                    lines.append(f"  {x['label']}: ${self.fmt_price(px)} ({x['explanation']})")

        return "\n".join(lines)

    def display_final_report(self, message=""):
        """
        Display the final report or show an error message in the result cards.
        """
        try:
            self.screen = self.sm.get_screen("main_screen")
            # screen.recommendation_results.text = f"Searching for '{search_query.upper()}'..."
            coin_name = self.screen.coin_input.text.strip()
            if not coin_name:
                raise ValueError("No coin name provided.")
            cid = self.search_coin_id(coin_name)
            if not cid:
                raise ValueError(f"No coin found for '{coin_name}'.")
            self.coin_id = cid

        except Exception as e:
            self.show_error(str(e))

    def parse_final_report(self):
        """
        Parse the formatted report into structured sections, including Entry and Exit points.
        """
        report_text = self.format_final_report()
        sections = {}
        current_section = "General Info"
        sections[current_section] = []

        # Flags for detecting entry and exit sections
        is_entry_section = False
        is_exit_section = False

        for line in report_text.split("\n"):
            line = line.strip()

            # Detect section transitions
            if "--- FUTURE TRADE IDEA ---" in line:
                current_section = "Future Trade"
                sections[current_section] = []
                is_entry_section = False
                is_exit_section = False

            elif "Multiple Entry Points:" in line:
                current_section = "Entry Points"
                sections[current_section] = []
                is_entry_section = True
                is_exit_section = False

            elif "Multiple Exit Points:" in line:
                current_section = "Exit Points"
                sections[current_section] = []
                is_exit_section = True
                is_entry_section = False

            # Append relevant lines to appropriate sections
            elif is_entry_section and line.startswith("  "):
                sections["Entry Points"].append(line)

            elif is_exit_section and line.startswith("  "):
                sections["Exit Points"].append(line)

            # General parsing for other sections
            elif ":" in line:
                if current_section not in sections:
                    sections[current_section] = []
                sections[current_section].append(line)

        # Handle cases when sections may not be populated
        if "Entry Points" not in sections:
            sections["Entry Points"] = ["No Entry Points found."]
        if "Exit Points" not in sections:
            sections["Exit Points"] = ["No Exit Points found."]

        # Return the sections as structured data
        return sections

    def create_result_cards(self, report_text):
        try:
            self.screen = self.sm.get_screen("main_screen")
            # Clear existing widgets
            result_container = self.screen.result_card_container
            result_container.clear_widgets()

            # Scrollable View with a horizontal box layout for cards
            scroll_view = MDScrollView(
                size_hint=(1, None),
                height=dp(350),
                scroll_type=['bars', 'content'],
                do_scroll_y= False
            )

            card_layout = MDBoxLayout(
                orientation='horizontal',
                spacing=dp(15),
                size_hint=(None, None),
                padding=(dp(15), dp(15)),
            )
            card_layout.bind(minimum_width=card_layout.setter('width'))
            scroll_view.add_widget(card_layout)

            # Parse report text into sections
            sections = self.parse_final_report()
            colors = ["#00a2a3", "#ee82ee", "#237e47", "#af0000"]

            # Create MD cards for each section
            for idx, (section_title, content_lines) in enumerate(sections.items()):
                content = "\n".join(content_lines)
                color = colors[idx % len(colors)]
                card = self.create_card(section_title, content, bg_color=color)
                card_layout.add_widget(card)

            # Set the width dynamically based on the number of cards
            card_layout.size_hint_y = None
            card_layout.height = dp(300)
            card_layout.width = dp(320) * len(sections)

            # Add the scrollable container to the UI
            result_container.add_widget(scroll_view)

        except Exception as e:
            self.show_error(f"Error creating result cards: {str(e)}")

    def show_error(self, error_message):
        """
        Display an error dialog to the user.
        """
        def error(*args):
            dialog = MDDialog(
                text=error_message,
                buttons=[
                    MDFlatButton(
                        text="OK",
                        on_release=lambda x: dialog.dismiss()
                    )
                ]
            )
            dialog.open()

        Clock.schedule_once(error)

    def create_card(self, title, content, bg_color=(0.2, 0.6, 0.3, 1)):
        """
        Create a single MD card with title and scrollable content.
        """
        card = MDCard(
            orientation='vertical',
            padding=dp(15),
            size_hint=(None,1),  # Dynamic height but fixed width
            size=(dp(300), dp(280)),
            elevation=10,
            md_bg_color=bg_color,
            radius=[dp(20)]
        )
        # Title
        title_label = MDLabel(
            text=title,
            theme_text_color="Custom",
            text_color=(1, 1, 1, 1),
            halign="center",
            bold=True,
            font_style="H4",
            size_hint_y=None,
            height=dp(30),
        )
        # Container for better content structure
        content_layout = MDBoxLayout(
            orientation='vertical',
            size_hint= (None,0.1),
            padding=(dp(3), dp(3)),
            spacing=dp(4)
        )
        # Scrollable container for text content
        scroll = MDScrollView(
            size_hint=(1, 1),  # Fully scrollable content within the card
            bar_width=dp(10),
        )

        # Content Label
        content_label = MDLabel(
            text=content.strip(),
            theme_text_color="Custom",
            text_color=(1,1,1,1),
            halign="left",
            valign="top",
        )
        content_label.bind(texture_size=content_label.setter('size'))


        # Add title, spacing, and scrollable text to the card
        scroll.add_widget(content_label)
        card.add_widget(title_label)
        card.add_widget(MDSeparator(height=dp(5), size_hint_y=.5))  # Optional separator for cleaner look
        card.add_widget(content_layout)
        card.add_widget(scroll)
        return card

    def fmt_price(self, val, decimals_if_large=2):
        """
        Format a price with dynamic decimal places:
         - if val >= 0.01 => 2 decimals
         - else => up to 8 decimals
        For market cap or big numbers, we do 2 decimals by default.
        """
        if val is None or np.isnan(val):
            return "N/A"
        if val >= 0.01:
            return f"{val:,.{decimals_if_large}f}"
        else:
            # up to 8 decimals for small
            return f"{val:.8f}"

    # -------------- UI Helpers --------------

    @mainthread
    def update_progress(self, value):
        """
        Safely update progress bar & label in the dialog from the background thread.
        """
        if not hasattr(self, 'progress_dialog') or not self.progress_dialog:
            return
        c = self.progress_dialog.content_cls
        if not c:
            return
        pb = c.ids.progress_bar
        lbl= c.ids.progress_label
        pb.value = value
        lbl.text = f"{int(value)}%"

    @mainthread
    def set_result_label(self, text):
        self.create_result_cards(text)

    @mainthread
    def dismiss_dialog(self):
        """
        Close the progress dialog from the main thread.
        """
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.dismiss()
            self.progress_dialog = None

    def run_analysiss(self):
        """Called when the user presses 'Run Analysis'."""
        self.loading_content = LoadingDialog()
        self.progress_dialog = MDDialog(
            title="Running Analysis",
            type="custom",
            content_cls=self.loading_content,
            auto_dismiss=False
        )
        self.progress_dialog.open()

        # Start the background thread
        thread = threading.Thread(
            target=self._run_main_in_background, daemon=True
        )
        thread.start()

    def _run_main_in_background(self):
        """
        Runs main() in a thread. We pass a callback so main() can
        report progress.
        """

        def update_ui_progress(value):
            # Schedule on the main thread
            Clock.schedule_once(lambda dt: self._update_progress(value))

        try:
            final_message = mainfoo(update_ui_progress)
        except Exception as e:
            logging.exception("Error in main()")
            final_message = f"[color=ff0000]Error:[/color] {str(e)}"

        # Once done, schedule the results dialog
        Clock.schedule_once(lambda dt: self._analysis_finished(final_message))

    def _update_progress(self, value):
        """Update the progress bar's value (0..100)."""
        if self.loading_content:
            pb = self.loading_content.ids.progress_bar
            pb.value = value
            label = self.loading_content.ids.label_progress
            label.text = f"{int(value)}%"

    def _analysis_finished(self, final_message):
        if self.progress_dialog:
            self.progress_dialog.dismiss(force=True)

        content = ResultsDialogContent(message=final_message)

        self.results_dialog = MDDialog(
            title="Analysis Results",
            type="custom",
            content_cls=content,
            #size_hint=(0.9, 0.9),  # <--- Make the dialog 90% of window size
            auto_dismiss=True
        )
        self.results_dialog.open()

    def show_alert_dialog(self):
        if not self.dialog:
            self.dialog = MDDialog(
                text="First Search any coin to show its result in terminal !  !  !",
            )
        self.dialog.open()

    def show_android_toast(self, message):
        try:
            print("Attempting to show toast:", message)
            # Get Android classes
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            Toast = autoclass('android.widget.Toast')
            String = autoclass('java.lang.String')

            activity = PythonActivity.mActivity
            java_message = String(message)

            # Create the toast object
            toast_obj = Toast.makeText(activity, java_message, Toast.LENGTH_SHORT)

            # Use a proper Runnable to show the toast on the UI thread
            runnable = ToastRunnable(toast_obj)
            activity.runOnUiThread(runnable)
        except Exception as e:
            print("Error showing toast:", e)

    def update_cards(self):
        self.screen = self.sm.get_screen("mfunc")
        COIN_NAME = self.screen.coin_input.text.strip()
        if not COIN_NAME:
            self.screen.header_label.text = "Please enter a coin symbol."
            return

        # Record the start time for minimum dialog display duration.
        self.analysis_start_time = time.time()

        # Show the progress dialog.
        self.show_progress_dialog()

        # Run the analysis in a background thread.
        future = executor.submit(self.run_analysis, COIN_NAME)
        # When the future is done, schedule UI update on main thread.
        future.add_done_callback(lambda fut: Clock.schedule_once(lambda dt: self.update_ui(fut.result()), 0))

    def run_analysis(self, COIN_NAME):
        try:
            # Run your heavy analysis (this runs in the background)
            analysis = main2(COIN_NAME)
        except Exception as e:
            analysis = {"error": str(e)}
        return analysis

    @mainthread
    def update_ui(self, analysis):
        self.screen = self.sm.get_screen("mfunc")
        if "error" in analysis:
            self.screen.header_label.text = f"Error: {analysis['error']}"
        else:
            self.screen.header_label.text = (
                f"Coin: {analysis['Coin'].title()} ({analysis['coin_id']})\n"
                f"Latest Price: {analysis['Price']} USD"
            )
            try:
                reddit_sent = reddit_sentiment_analysis(self.screen.coin_input.text.strip())
            except Exception as e:
                reddit_sent = f"Reddit Sentiment Error: {str(e)}"
            self.screen.reddit_label.text = reddit_sent

            self.screen.indicators_label.text = (
                f"[b]EMA (20 period):[/b] {analysis['EMA']}  [color=#00FF00]{analysis['EMA_Signal']}[/color]\n"
                f"[b]RSI (14 period):[/b] {analysis['RSI']}  [color=#FFD700]{analysis['RSI_Signal']}[/color]\n"
                f"[b]ADX (14 period):[/b] {analysis['ADX']}  Trend: [color=#00BFFF]{analysis['ADX_Signal']}[/color]\n"
                f"[b]MACD:[/b] {analysis['MACD']} vs Signal: {analysis['MACD_Signal']}  Filter: [color=#FF4500]{analysis['MACD_Signal']}[/color]\n"
                f"[b]Candlestick Signal:[/b] {analysis['Candlestick']}\n"
                f"[b]SMC Signal:[/b] {analysis['SMC']} (High: {analysis['Recent_High']}, Low: {analysis['Recent_Low']})"
            )

            self.screen.fibonacci_label.text = analysis['Fibonacci']
            self.screen.fib_signal_label.text = f"Fibonacci Signal: {analysis['Fibonacci_Signal']}"
            self.screen.volume_label.text = (
                f"Latest Volume: {analysis['Volume']}\n"
                f"Volume SMA (20 period): {analysis['Volume_SMA']}\n"
                f"Volume Signal: {analysis['Volume_Signal']}"
            )
            self.screen.votes_label.text = (
                f"Votes -> Bullish: {analysis['Bullish_Votes']}, Bearish: {analysis['Bearish_Votes']}\n"
                f"Final Combined Trend: {analysis['Final_Trend']}"
            )
            self.screen.reason_label.text = analysis['Reason_Summary']

        # Once analysis is done, if the progress bar is not complete, force it to 100 and wait briefly.
        if self.progress_bar.value < 100:
            self.progress_bar.value = 100
            Clock.schedule_once(lambda dt: self.stop_progress_dialog(), 0.5)
        else:
            self.stop_progress_dialog()

    def show_progress_dialog(self):
        from kivymd.uix.dialog import MDDialog
        from kivymd.uix.boxlayout import MDBoxLayout
        from kivy.metrics import dp
        print("Opening progress dialog")
        if not self.dialog:
            content = MDBoxLayout(orientation="vertical", size_hint=(1, None), height=dp(50))
            self.progress_bar = MDProgressBar(value=0, size_hint=(1, None), height=dp(10))
            content.add_widget(self.progress_bar)
            self.dialog = MDDialog(
                title="Processing...",
                type="custom",
                content_cls=content,
                auto_dismiss=False,
                size_hint=(0.8, None),
                height=dp(150)
            )
        self.dialog.open()
        self.start_progress()

    def start_progress(self):
        """Initialize and start the determinate progress bar."""
        self.progress_bar.value = 0
        self.progress_event = Clock.schedule_interval(self.increment_progress, 0.1)

    def increment_progress(self, dt):
        """Increment the progress bar's value until 100."""
        pb = self.progress_bar
        if pb.value < 100:
            pb.value += 0.5  # Increase progress by 2% every 0.1 second (will take ~5 seconds to reach 100)
        # If pb.value >= 100, we simply do nothing here.

    def stop_progress_dialog(self):
        """Stop updating the progress bar and dismiss the dialog."""
        if hasattr(self, 'progress_event'):
            self.progress_event.cancel()
        if self.dialog:
            self.dialog.dismiss()
            self.dialog = None

if __name__ == "__main__":
    CryptoCoachingApp().run()

