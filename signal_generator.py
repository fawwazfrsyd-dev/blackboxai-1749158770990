import numpy as np
from data_fetcher import get_current_price, get_current_volume, historical_data
from technical_analysis import calculate_indicators
from ml_models import predict_price, predict_volatility, cluster_patterns, detect_anomaly, train_rl, rl_recommendation, train_dqn, dqn_trailing_stop
from fundamental_analysis import scrape_news, analyze_news_sentiment, scrape_economic_calendar, scrape_cot_report, scrape_x_sentiment, calculate_sentiment_heatmap, calculate_correlation_with_lag, get_vix_sentiment, speech_to_text_analysis
from risk_management import dynamic_risk, optimize_risk_reward, suggest_hedging, monte_carlo_simulation
from order_flow import calculate_order_flow, calculate_market_profile
from config import CURRENCY_PAIR
from datetime import datetime, timedelta
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import torch
from hmmlearn import hmm
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Cache
indicators_cache = {}
fundamentals_cache = {}
transformer_prediction_cache = {}
gnn_prediction_cache = {}
cache_timeout = timedelta(minutes=5)

# Inisialisasi model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

class GNNModel(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_dim=16, output_dim=8):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.linear = torch.nn.Linear(output_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.linear(x)

gnn_model = GNNModel()
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)

def get_cached_indicators(pair, volumes):
    current_time = datetime.now()
    cache_key = f"{pair}_{len(volumes)}"
    if cache_key in indicators_cache and (current_time - indicators_cache[cache_key][1]) < cache_timeout:
        return indicators_cache[cache_key][0]
    indicators = calculate_indicators(historical_data[pair], volumes)
    indicators_cache[cache_key] = (indicators, current_time)
    return indicators

def get_cached_fundamentals():
    current_time = datetime.now()
    if 'fundamentals' in fundamentals_cache and (current_time - fundamentals_cache['fundamentals'][1]) < cache_timeout:
        return fundamentals_cache['fundamentals'][0]
    fundamentals = {
        'news': scrape_news(),
        'news_sentiment': analyze_news_sentiment(scrape_news()),
        'x_sentiment': scrape_x_sentiment(),
        'sentiment_heatmap': calculate_sentiment_heatmap(analyze_news_sentiment(scrape_news()), scrape_x_sentiment()),
        'correlation': calculate_correlation_with_lag(),
        'econ_events': scrape_economic_calendar(),
        'cot_position': scrape_cot_report(),
        'vix_sentiment': get_vix_sentiment(),
        'audio_sentiment': speech_to_text_analysis()
    }
    fundamentals_cache['fundamentals'] = (fundamentals, current_time)
    return fundamentals

def transformer_price_prediction(data):
    current_time = datetime.now()
    cache_key = f"transformer_pred_{len(data)}"
    if cache_key in transformer_prediction_cache and (current_time - transformer_prediction_cache[cache_key][1]) < cache_timeout:
        return transformer_prediction_cache[cache_key][0]

    data_array = np.array(data[-100:]).reshape(1, -1, 1)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(100, 1)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(data_array, data_array, epochs=1, verbose=0)
    prediction = model.predict(data_array)[-1][0]

    fundamentals = get_cached_fundamentals()
    sentiment_score = (fundamentals['news_sentiment'] + fundamentals['x_sentiment'] + fundamentals['audio_sentiment']) / 3
    prediction += sentiment_score * 0.1

    transformer_prediction_cache[cache_key] = (prediction, current_time)
    return prediction

def gnn_price_prediction(data):
    current_time = datetime.now()
    cache_key = f"gnn_pred_{len(data)}"
    if cache_key in gnn_prediction_cache and (current_time - gnn_prediction_cache[cache_key][1]) < cache_timeout:
        return gnn_prediction_cache[cache_key][0]

    G = nx.Graph()
    assets = ['XAUUSD', 'DXY', 'EURUSD', 'USDJPY']
    correlations = get_cached_fundamentals()['correlation']
    for i, asset1 in enumerate(assets):
        G.add_node(asset1, feature=np.array([data[-1] if asset1 == 'XAUUSD' else 0, correlations.get(asset1, 0), 1]))
        for j, asset2 in enumerate(assets[i+1:], i+1):
            if correlations.get(asset1, {}).get(asset2, 0) != 0:
                G.add_edge(asset1, asset2, weight=correlations.get(asset1, {}).get(asset2, 0))

    edge_index = torch.tensor([[i, j] for i, j, _ in G.edges(data=True)], dtype=torch.long).t()
    x = torch.tensor([data[0] for _, data in G.nodes(data=True)], dtype=torch.float).reshape(-1, 3)
    data_gnn = Data(x=x, edge_index=edge_index)

    gnn_model.train()
    optimizer.zero_grad()
    out = gnn_model(data_gnn)
    loss = torch.nn.functional.mse_loss(out, torch.tensor([data[-1]], dtype=torch.float))
    loss.backward()
    optimizer.step()
    prediction = out.detach().numpy()[0][0]

    gnn_prediction_cache[cache_key] = (prediction, current_time)
    return prediction

def quantum_inspired_optimization(data, current_price, volatility):
    np.random.seed(int(datetime.now().timestamp()))
    iterations = 50  # Dioptimalkan untuk kecepatan
    best_levels = {'sl': current_price, 'tp1': current_price, 'tp2': current_price, 'tp3': current_price}
    best_reward = -float('inf')

    for _ in range(iterations):
        sl_offset = np.random.normal(0, volatility * 0.1)
        tp1_offset = np.random.normal(volatility * 0.5, volatility * 0.2)
        tp2_offset = np.random.normal(volatility * 1.0, volatility * 0.3)
        tp3_offset = np.random.normal(volatility * 1.5, volatility * 0.4)

        sl = current_price - abs(sl_offset) if np.random.rand() < 0.5 else current_price + abs(sl_offset)
        tp1 = current_price + abs(tp1_offset) if np.random.rand() < 0.5 else current_price - abs(tp1_offset)
        tp2 = current_price + abs(tp2_offset) if np.random.rand() < 0.5 else current_price - abs(tp2_offset)
        tp3 = current_price + abs(tp3_offset) if np.random.rand() < 0.5 else current_price - abs(tp3_offset)

        future_prices = data[-50:]
        reward = 0
        for price in future_prices:
            if price >= max(tp1, tp2, tp3) or price <= sl:
                reward += (max(tp1, tp2, tp3) - current_price) if price >= max(tp1, tp2, tp3) else (sl - current_price)
                break
        if reward > best_reward:
            best_reward = reward
            best_levels = {'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3}

    return best_levels['sl'], best_levels['tp1'], best_levels['tp2'], best_levels['tp3']

def detect_market_regime(data):
    model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=50)  # Dioptimalkan iterasi
    data_array = np.array(data[-100:]).reshape(-1, 1)
    model.fit(data_array)
    hidden_states = model.predict(data_array)
    regime = np.bincount(hidden_states[-10:]).argmax()
    return 'trending' if regime == 0 else 'ranging' if regime == 1 else 'volatile'

def backtest_strategy(data, timeframe, strategy):
    signals = [generate_signal(0, timeframe, strategy, data[i-100:i]) for i in range(100, len(data) - 50) if generate_signal(0, timeframe, strategy, data[i-100:i])]
    if not signals:
        return {'win_rate': 0, 'total_profit': 0, 'max_drawdown': 0, 'avg_confidence': 0}

    wins, total_profit, balance, peak = 0, 0, 10000, 10000
    max_drawdown = 0
    for signal, i in zip(signals, range(100, len(data) - 50)):
        future_prices = data[i:i+50]
        entry_price = signal['price']
        stop_loss = signal['stop_loss']
        take_profit_1 = signal['take_profit_1']
        for price in future_prices:
            if signal['recommendation'] == 'BUY':
                if price >= take_profit_1:
                    profit = (take_profit_1 - entry_price) * signal['position_size']
                    total_profit += profit
                    wins += 1
                    break
                elif price <= stop_loss:
                    profit = (stop_loss - entry_price) * signal['position_size']
                    total_profit += profit
                    break
            else:
                if price <= take_profit_1:
                    profit = (entry_price - take_profit_1) * signal['position_size']
                    total_profit += profit
                    wins += 1
                    break
                elif price >= stop_loss:
                    profit = (entry_price - stop_loss) * signal['position_size']
                    total_profit += profit
                    break
        balance += profit
        drawdown = (peak - balance) / peak * 100
        max_drawdown = max(max_drawdown, drawdown)
        peak = max(peak, balance)

    return {
        'win_rate': (wins / len(signals)) * 100,
        'total_profit': total_profit,
        'max_drawdown': max_drawdown,
        'avg_confidence': sum(s['confidence_score'] for s in signals) / len(signals)
    }

def generate_signal(chat_id, timeframe='15', strategy='intraday', sim_data=None):
    current_price = get_current_price()
    current_volume = get_current_volume()
    data_to_use = sim_data if sim_data is not None else historical_data[CURRENCY_PAIR]
    if not data_to_use or len(data_to_use) < 50 or (current_price == 0.0 and sim_data is None) or (current_volume == 0.0 and sim_data is None):
        return None

    volumes = [get_current_volume() for _ in range(len(data_to_use))] if sim_data is None else [1] * len(data_to_use)
    indicators = get_cached_indicators(CURRENCY_PAIR, volumes)
    volatility = predict_volatility(data_to_use)
    cluster = cluster_patterns(data_to_use)
    anomaly = detect_anomaly(data_to_use, model_type='deep_learning')
    order_flow = calculate_order_flow(volumes)
    va_low, va_high = calculate_market_profile(data_to_use)
    
    predicted_price = gnn_price_prediction(data_to_use)  # Prioritaskan GNN untuk akurasi

    fundamentals = get_cached_fundamentals()
    news_sentiment, x_sentiment, audio_sentiment = fundamentals['news_sentiment'], fundamentals['x_sentiment'], fundamentals['audio_sentiment']
    sentiment_heatmap = fundamentals['sentiment_heatmap']
    correlation, econ_events, cot_position, vix_sentiment = fundamentals['correlation'], fundamentals['econ_events'], fundamentals['cot_position'], fundamentals['vix_sentiment']

    tf_4h = get_cached_indicators(CURRENCY_PAIR, volumes[-400:]) if len(data_to_use) >= 400 else {}
    tf_daily = get_cached_indicators(CURRENCY_PAIR, volumes[-960:]) if len(data_to_use) >= 960 else {}
    mtf_confirmation = (tf_4h.get('ma_short', 0) > tf_4h.get('ma_long', 0) and tf_daily.get('ma_short', 0) > tf_daily.get('ma_long', 0)) or \
                      (tf_4h.get('ma_short', 0) < tf_4h.get('ma_long', 0) and tf_daily.get('ma_short', 0) < tf_daily.get('ma_long', 0))

    q_table = train_rl(data_to_use, dynamic=True)
    rl_action = rl_recommendation(data_to_use, q_table)
    market_regime = detect_market_regime(data_to_use)

    trend = 'BULLISH' if predicted_price > current_price else 'BEARISH'
    bias = 'BULLISH' if all([
        indicators['rsi'] < 30, indicators['macd'] > indicators['signal_line'], indicators['tenkan'] > indicators['kijun'],
        indicators['stoch_k'] < 20, indicators['adx'] > 25, indicators['cci'] < -100, indicators['sar'] < current_price,
        current_price > indicators['vwap'], indicators['vsa_signal'] == "Buying Pressure", indicators['fractal'] == "Bullish Fractal",
        indicators['renko_trend'] == "Uptrend", indicators['mmo'] > 0, indicators['vwm'] > 0, current_price > indicators['keltner_lower'],
        indicators['elder_ray_bull'] > 0, indicators['cvd'] > 0, indicators['trend'] > 0, indicators['market_breadth'] > 0.6,
        current_price > indicators['wavelet_trend'], mtf_confirmation, rl_action == "BUY", indicators['volatility_breakout']
    ]) else 'BEARISH' if all([
        indicators['rsi'] > 70, indicators['macd'] < indicators['signal_line'], indicators['tenkan'] < indicators['kijun'],
        indicators['stoch_k'] > 80, indicators['adx'] > 25, indicators['cci'] > 100, indicators['sar'] > current_price,
        current_price < indicators['vwap'], indicators['vsa_signal'] == "Selling Pressure", indicators['fractal'] == "Bearish Fractal",
        indicators['renko_trend'] == "Downtrend", indicators['mmo'] < 0, indicators['vwm'] < 0, current_price < indicators['keltner_upper'],
        indicators['elder_ray_bear'] < 0, indicators['cvd'] < 0, indicators['trend'] < 0, indicators['market_breadth'] < 0.4,
        current_price < indicators['wavelet_trend'], mtf_confirmation, rl_action == "SELL", indicators['volatility_breakout']
    ]) else trend

    if sentiment_heatmap > 0.1 or cot_position > 0 or audio_sentiment > 0:
        bias = 'BULLISH'
    elif sentiment_heatmap < -0.1 or cot_position < 0 or audio_sentiment < 0:
        bias = 'BEARISH'
    if econ_events and any('FOMC' in e['event'] or 'NFP' in e['event'] for e in econ_events):
        return None
    if anomaly:
        return None
    
    recommendation = 'BUY' if bias == 'BULLISH' else 'SELL'
    if strategy == 'mean_reversion' and indicators['rsi'] > 70 and current_price < indicators['lower_band']:
        recommendation = 'BUY'
    elif strategy == 'mean_reversion' and indicators['rsi'] < 30 and current_price > indicators['upper_band']:
        recommendation = 'SELL'

    confidence_score = 50 + (15 if indicators['rsi'] < 30 or indicators['rsi'] > 70 else 0) + \
                       (10 if abs(indicators['histogram']) > 0.5 else 0) + (10 if indicators['adx'] > 25 else 0) + \
                       (10 if abs(indicators['cci']) > 100 else 0) + (10 if indicators['candlestick_pattern'] != "None" else 0) + \
                       (10 if indicators['fractal'] != "None" else 0) + (5 if indicators['mmo'] != 0 else 0) + \
                       (5 if indicators['renko_trend'] != "None" else 0) + (5 if indicators['vwm'] != 0 else 0) + \
                       (5 if indicators['cvd'] != 0 else 0) + (sentiment_heatmap + x_sentiment + audio_sentiment) * 5
    confidence = 'HIGH' if confidence_score >= 80 else 'MEDIUM' if confidence_score >= 70 else 'LOW'

    stop_loss, take_profit_1, take_profit_2, take_profit_3 = quantum_inspired_optimization(data_to_use, current_price, volatility)
    dqn_model = train_dqn(data_to_use)
    atr_trailing_stop = dqn_trailing_stop(dqn_model, current_price, volatility, market_regime)

    take_profit_1, rr_ratio_1, _ = optimize_risk_reward(current_price, stop_loss, take_profit_1, indicators['fib_levels'], indicators['pivot_points'], indicators['murrey_lines'], indicators['paz'], volatility)
    take_profit_2, rr_ratio_2, _ = optimize_risk_reward(current_price, stop_loss, take_profit_2, indicators['fib_levels'], indicators['pivot_points'], indicators['murrey_lines'], indicators['paz'], volatility)
    take_profit_3, rr_ratio_3, _ = optimize_risk_reward(current_price, stop_loss, take_profit_3, indicators['fib_levels'], indicators['pivot_points'], indicators['murrey_lines'], indicators['paz'], volatility)

    rr_ratio = (rr_ratio_1 + rr_ratio_2 + rr_ratio_3) / 3
    position_size = dynamic_risk(current_price, indicators['atr'], volatility, strategy, return_size=True)
    hedging = suggest_hedging(correlation, bias)
    expected_return, risk_std = monte_carlo_simulation({
        'recommendation': recommendation, 'take_profit': take_profit_1, 'stop_loss': stop_loss,
        'predicted_price': predicted_price, 'volatility': volatility
    }, data_to_use)

    if market_regime == 'trending':
        strategy = 'swing'
    elif market_regime == 'ranging':
        strategy = 'mean_reversion'
    elif market_regime == 'volatile':
        strategy = 'scalping'

    return {
        'price': current_price, 'confidence': confidence, 'confidence_score': min(confidence_score, 100),
        'bias': bias, 'recommendation': recommendation, 'stop_loss': stop_loss, 'take_profit_1': take_profit_1,
        'take_profit_2': take_profit_2, 'take_profit_3': take_profit_3, 'rr_ratio_1': rr_ratio_1, 'rr_ratio_2': rr_ratio_2,
        'rr_ratio_3': rr_ratio_3, 'rr_ratio': rr_ratio, 'position_size': position_size, 'rsi': indicators['rsi'],
        'macd_histogram': indicators['histogram'], 'stoch_k': indicators['stoch_k'], 'stoch_d': indicators['stoch_d'],
        'adx': indicators['adx'], 'cci': indicators['cci'], 'sar': indicators['sar'], 'vwap': indicators['vwap'],
        'volatility': volatility, 'predicted_price': predicted_price, 'cluster': cluster, 'anomaly': anomaly,
        'news_sentiment': news_sentiment, 'x_sentiment': x_sentiment, 'sentiment_heatmap': sentiment_heatmap,
        'correlation': correlation, 'hedging': hedging, 'strategy': strategy, 'timeframe': timeframe,
        'fib_levels': indicators['fib_levels'], 'ichimoku': (indicators['tenkan'], indicators['kijun'], indicators['senkou_a'], indicators['senkou_b']),
        'heikin_ashi': indicators['heikin_ashi'], 'candlestick_pattern': indicators['candlestick_pattern'],
        'elliott_wave': indicators['elliott_wave'], 'harmonic_pattern': indicators['harmonic_pattern'],
        'gann_angles': indicators['gann_angles'], 'cycle_period': indicators['cycle_period'], 'fractal': indicators['fractal'],
        'vsa_signal': indicators['vsa_signal'], 'market_delta': indicators['market_delta'], 'order_flow': order_flow,
        'va_low': va_low, 'va_high': va_high, 'pivot_points': indicators['pivot_points'], 'murrey_lines': indicators['murrey_lines'],
        'seasonality': indicators['seasonality'], 'wavelet_trend': indicators['wavelet_trend'], 'mmo': indicators['mmo'],
        'paz': indicators['paz'], 'atr_trailing_stop': atr_trailing_stop, 'renko_trend': indicators['renko_trend'],
        'dvi': indicators['dvi'], 'keltner_upper': indicators['keltner_upper'], 'keltner_lower': indicators['keltner_lower'],
        'elder_ray_bull': indicators['elder_ray_bull'], 'elder_ray_bear': indicators['elder_ray_bear'], 'cvd': indicators['cvd'],
        'trend': indicators['trend'], 'seasonal': indicators['seasonal'], 'residual': indicators['residual'],
        'market_breadth': indicators['market_breadth'], 'vwm': indicators['vwm'], 'volatility_breakout': indicators['volatility_breakout'],
        'cot_position': cot_position, 'vix_sentiment': vix_sentiment, 'econ_events': econ_events, 'audio_sentiment': audio_sentiment,
        'expected_return': expected_return, 'risk_std': risk_std, 'market_regime': market_regime
    }
