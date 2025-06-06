import numpy as np

def calculate_order_flow(volumes):
    return np.sum(volumes[-10:]) - np.sum(volumes[-20:-10])  # Perbedaan volume terakhir

def calculate_market_profile(data):
    prices = np.array(data[-100:])
    volume = np.ones(len(prices))  # Placeholder, ganti dengan data volume nyata
    hist, bin_edges = np.histogram(prices, bins=10, weights=volume)
    va_low = bin_edges[np.argmax(hist) - 1]
    va_high = bin_edges[np.argmax(hist) + 1]
    return va_low, va_high
