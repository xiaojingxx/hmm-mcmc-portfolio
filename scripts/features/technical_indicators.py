def momentum(prices, window=20):
    return prices.pct_change(window)
