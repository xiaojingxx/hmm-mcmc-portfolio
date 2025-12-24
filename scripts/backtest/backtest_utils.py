import numpy as np
import pandas as pd

def backtest_portfolio(returns, freq=252):
    """
    Compute standard backtest metrics.
    """
    returns = returns.dropna()

    cumulative_return = (1 + returns).prod() - 1
    annual_return = (1 + cumulative_return) ** (freq / len(returns)) - 1
    annual_vol = returns.std() * np.sqrt(freq)
    sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan
    max_dd = max_drawdown(returns)

    return {
        "Annual Return": annual_return,
        "Annual Volatility": annual_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd
    }


def max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()
