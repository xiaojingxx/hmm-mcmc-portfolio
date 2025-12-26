import numpy as np
import pandas as pd

def compute_log_returns(prices):
    prices = prices.sort_index()
    log_returns = np.log(prices / prices.shift(1)).fillna(0)

    return log_returns

    

