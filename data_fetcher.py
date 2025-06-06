import websocket
import json
import requests
from config import TRADERMADE_API_KEY, TRADERMADE_WEBSOCKET_URL, TRADERMADE_BASE_URL, CURRENCY_PAIR, DATA_TIMEFRAME
from datetime import datetime, timedelta
import pandas as pd
import threading

# Cache data historis
historical_data = {CURRENCY_PAIR: []}
current_price = 0.0
current_volume = 0.0
lock = threading.Lock()

def on_message(ws, message):
    global current_price, current_volume
    try:
        data = json.loads(message)
        if data['symbol'] == CURRENCY_PAIR:
            with lock:
                current_price = float(data.get('mid', current_price))
                current_volume = float(data.get('volume', current_volume))
    except Exception as e:
        print(f"WebSocket Error: {e}")

def on_error(ws, error):
    print(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket Closed")

def on_open(ws):
    print("WebSocket Opened")
    ws.send(json.dumps({"userKey": TRADERMADE_API_KEY, "symbol": CURRENCY_PAIR}))

def start_websocket():
    ws = websocket.WebSocketApp(
        TRADERMADE_WEBSOCKET_URL,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    threading.Thread(target=ws.run_forever, daemon=True).start()

def get_current_price():
    with lock:
        return current_price

def get_current_volume():
    with lock:
        return current_volume

def fetch_historical_data(symbol=CURRENCY_PAIR, period=500, timeframe=DATA_TIMEFRAME):
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=period * int(timeframe[:-1]))
        params = {
            'currency': symbol,
            'start': start_time.strftime('%Y-%m-%d-%H:%M'),
            'end': end_time.strftime('%Y-%m-%d-%H:%M'),
            'period': timeframe,
            'api_key': TRADERMADE_API_KEY
        }
        response = requests.get(f"{TRADERMADE_BASE_URL}/timeseries", params=params)
        data = response.json()
        df = pd.DataFrame(data['quotes'], columns=['date', 'open', 'high', 'low', 'close'])
        df['date'] = pd.to_datetime(df['date'])
        historical_data[symbol] = df['close'].tolist()
        return historical_data[symbol]
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return []

# Inisialisasi WebSocket dan data historis
start_websocket()
historical_data[CURRENCY_PAIR] = fetch_historical_data()
