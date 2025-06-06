import numpy as np
from config import RISK_PER_TRADE

def dynamic_risk(current_price, atr, volatility, strategy, return_size=False):
    risk_per_trade = RISK_PER_TRADE
    if strategy == 'scalping':
        risk_amount = current_price * atr * 0.5
    elif strategy == 'intraday':
        risk_amount = current_price * atr
    else:  # swing
        risk_amount = current_price * atr * 2
    position_size = (risk_per_trade * 10000) / (volatility * atr) if volatility > 0 else 0.1
    return position_size if return_size else risk_amount

def optimize_risk_reward(entry_price, stop_loss, take_profit, fib_levels, pivot_points, murrey_lines, paz, volatility):
    distance_to_sl = abs(entry_price - stop_loss)
    distance_to_tp = abs(entry_price - take_profit)
    rr_ratio = distance_to_tp / distance_to_sl if distance_to_sl > 0 else 1.0
    adjusted_tp = take_profit + (volatility * 0.1) if entry_price < take_profit else take_profit - (volatility * 0.1)
    return adjusted_tp, rr_ratio, distance_to_sl

def suggest_hedging(correlation, bias):
    if correlation.get('DXY', 0) > 0.5 and bias == 'BEARISH':
        return "Hedge with USD strength (e.g., USDJPY)"
    return "No hedging recommended"

def monte_carlo_simulation(params, data):
    trials = 1000
    returns = np.random.normal(params['predicted_price'], params['volatility'], trials)
    profits = [(p - params['take_profit']) if params['recommendation'] == 'SELL' else (params['take_profit'] - p) for p in returns if p <= params['stop_loss'] or p >= params['take_profit']]
    return np.mean(profits) if profits else 0.0, np.std(profits) if profits else params['volatility'] 
