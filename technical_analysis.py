import numpy as np
import talib
from data_fetcher import historical_data

def calculate_indicators(data, volumes):
    if len(data) < 100 or len(volumes) < 100:
        return {key: 0 for key in ['rsi', 'macd', 'signal_line', 'histogram', 'stoch_k', 'stoch_d', 'adx', 'cci', 'sar', 'vwap', 'atr',
                                   'tenkan', 'kijun', 'senkou_a', 'senkou_b', 'heikin_ashi', 'candlestick_pattern',
                                   'elliott_wave', 'harmonic_pattern', 'gann_angles', 'cycle_period', 'fractal',
                                   'vsa_signal', 'market_delta', 'pivot_points', 'murrey_lines', 'seasonality',
                                   'wavelet_trend', 'mmo', 'paz', 'renko_trend', 'dvi', 'keltner_upper', 'keltner_lower',
                                   'elder_ray_bull', 'elder_ray_bear', 'cvd', 'trend', 'seasonal', 'residual',
                                   'market_breadth', 'vwm', 'volatility_breakout', 'fib_levels', 'ma_short', 'ma_long', 'upper_band', 'lower_band']}

    close = np.array(data[-100:], dtype=float)
    high = np.array(data[-100:], dtype=float)  # Placeholder, ganti dengan data high nyata jika tersedia
    low = np.array(data[-100:], dtype=float)   # Placeholder, ganti dengan data low nyata jika tersedia
    volume = np.array(volumes[-100:], dtype=float)
    
    indicators = {}
    indicators['rsi'] = talib.RSI(close, timeperiod=14)[-1]
    macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    indicators['macd'] = macd[-1] if not np.isnan(macd[-1]) else 0
    indicators['signal_line'] = signal[-1] if not np.isnan(signal[-1]) else 0
    indicators['histogram'] = hist[-1] if not np.isnan(hist[-1]) else 0
    stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
    indicators['stoch_k'] = stoch_k[-1] if not np.isnan(stoch_k[-1]) else 0
    indicators['stoch_d'] = stoch_d[-1] if not np.isnan(stoch_d[-1]) else 0
    indicators['adx'] = talib.ADX(high, low, close, timeperiod=14)[-1] if not np.isnan(talib.ADX(high, low, close, timeperiod=14)[-1]) else 0
    indicators['cci'] = talib.CCI(high, low, close, timeperiod=14)[-1] if not np.isnan(talib.CCI(high, low, close, timeperiod=14)[-1]) else 0
    indicators['sar'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)[-1] if not np.isnan(talib.SAR(high, low, acceleration=0.02, maximum=0.2)[-1]) else close[-1]
    indicators['vwap'] = np.sum(close * volume) / np.sum(volume) if np.sum(volume) > 0 else close[-1]
    indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)[-1] if not np.isnan(talib.ATR(high, low, close, timeperiod=14)[-1]) else 0
    indicators['tenkan'] = (np.max(close[-9:]) + np.min(close[-9:])) / 2
    indicators['kijun'] = (np.max(close[-26:]) + np.min(close[-26:])) / 2
    indicators['senkou_a'] = (indicators['tenkan'] + indicators['kijun']) / 2
    indicators['senkou_b'] = (np.max(close[-52:]) + np.min(close[-52:])) / 2
    indicators['heikin_ashi'] = np.mean(close[-4:])
    indicators['candlestick_pattern'] = "None"  # Placeholder
    indicators['elliott_wave'] = "None"  # Placeholder
    indicators['harmonic_pattern'] = "None"  # Placeholder
    indicators['gann_angles'] = {'1x1': close[-1] + indicators['atr']}
    indicators['cycle_period'] = 14  # Placeholder
    indicators['fractal'] = "None"  # Placeholder
    indicators['vsa_signal'] = "None"  # Placeholder
    indicators['market_delta'] = np.sum(volume * np.sign(close - np.mean(close)))
    indicators['pivot_points'] = {'pivot': np.mean(close[-3:]), 's1': np.mean(close[-3:]) - indicators['atr'], 'r1': np.mean(close[-3:]) + indicators['atr']}
    indicators['murrey_lines'] = {'m1': close[-1]}  # Placeholder
    indicators['seasonality'] = np.mean(close[-30:]) - np.mean(close[-60:-30])  # Placeholder
    indicators['wavelet_trend'] = np.mean(close[-10:])  # Placeholder
    indicators['mmo'] = np.mean(close[-5:]) - np.mean(close[-10:-5])  # Placeholder
    indicators['paz'] = {'supply': max(close[-5:]), 'demand': min(close[-5:])}  # Placeholder
    indicators['renko_trend'] = "None"  # Placeholder
    indicators['dvi'] = indicators['adx'] / indicators['cci'] if indicators['cci'] != 0 else 0  # Placeholder
    indicators['keltner_upper'] = close[-1] + 2 * indicators['atr']
    indicators['keltner_lower'] = close[-1] - 2 * indicators['atr']
    indicators['elder_ray_bull'] = close[-1] - np.mean(close[-13:])  # Placeholder
    indicators['elder_ray_bear'] = np.mean(close[-13:]) - close[-1]  # Placeholder
    indicators['cvd'] = np.sum(volume[-10:]) - np.sum(volume[-20:-10])  # Placeholder
    indicators['trend'] = 1 if close[-1] > close[-2] else -1  # Placeholder
    indicators['seasonal'] = indicators['seasonality']  # Placeholder
    indicators['residual'] = close[-1] - np.mean(close)  # Placeholder
    indicators['market_breadth'] = 0.5  # Placeholder
    indicators['vwm'] = indicators['vwap'] * volume[-1] / np.sum(volume) if np.sum(volume) > 0 else 0  # Placeholder
    indicators['volatility_breakout'] = 1 if indicators['atr'] > np.mean(close[-14:]) * 0.01 else 0  # Placeholder
    indicators['fib_levels'] = {f'level_{i}': close[-1] * (1 + i * 0.236) for i in range(5)}  # Placeholder
    indicators['ma_short'] = talib.SMA(close, timeperiod=10)[-1] if not np.isnan(talib.SMA(close, timeperiod=10)[-1]) else 0
    indicators['ma_long'] = talib.SMA(close, timeperiod=20)[-1] if not np.isnan(talib.SMA(close, timeperiod=20)[-1]) else 0
    upper_band, _, lower_band = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    indicators['upper_band'] = upper_band[-1] if not np.isnan(upper_band[-1]) else 0
    indicators['lower_band'] = lower_band[-1] if not np.isnan(lower_band[-1]) else 0

    return indicators
