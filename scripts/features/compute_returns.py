import numpy as np
import pandas as pd

def compute_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()
