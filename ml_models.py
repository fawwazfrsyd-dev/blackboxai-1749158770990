import numpy as np
import tensorflow as tf
from collections import deque
import random

def predict_price(data):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(len(data[-100:]),)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(np.array(data[-100:]).reshape(1, -1), np.array([data[-1]]), epochs=1, verbose=0)
    return model.predict(np.array(data[-100:]).reshape(1, -1))[0][0]

def predict_volatility(data):
    return np.std(data[-20:])

def cluster_patterns(data):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(np.array(data[-100:]).reshape(-1, 1))
    return clusters[-1]

def detect_anomaly(data, model_type='deep_learning'):
    if model_type == 'deep_learning':
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(len(data[-100:]),)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        prediction = model.predict(np.array(data[-100:]).reshape(1, -1))[0][0]
        return prediction > 0.9
    return False

def train_rl(data, dynamic=False):
    q_table = {}
    for state in range(len(data[-100:])):
        q_table[state] = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    return q_table

def rl_recommendation(data, q_table):
    state = len(data) - 1
    return max(q_table.get(state, {'BUY': 0, 'SELL': 0, 'HOLD': 0}), key=q_table.get(state, {'BUY': 0, 'SELL': 0, 'HOLD': 0}).get)

def train_dqn(data):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(len(data[-100:]),)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def dqn_trailing_stop(model, current_price, volatility, market_regime):
    state = np.array([current_price, volatility, 1 if market_regime == 'trending' else 0])
    action = model.predict(state.reshape(1, -1))[0]
    return current_price - (volatility * 0.5 * action[0])  # Sederhana, perlu tuning
