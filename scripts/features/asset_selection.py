"""
asset_selection.py

Data-driven asset universe selection for HMM-based portfolio optimisation.

Pipeline:
    50 assets
      → Data quality filter
      → Regime sensitivity (HMM-based dispersion)
      → Correlation clustering (diversification)
      → Marginal portfolio contribution
      → Final 15 assets
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import warnings

warnings.filterwarnings("ignore")


# =====================================================
# Utility functions
# =====================================================

def annualized_sharpe(returns: pd.Series) -> float:
    """Compute annualized Sharpe ratio."""
    if returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(252)


def max_drawdown(cum_returns: pd.Series) -> float:
    """Compute max drawdown from cumulative returns."""
    return (cum_returns.cummax() - cum_returns).max()


def portfolio_returns(df: pd.DataFrame) -> pd.Series:
    """Equal-weight portfolio returns."""
    return df.mean(axis=1)


# =====================================================
# Step 1: Data quality filter
# =====================================================

def data_quality_filter(
    returns,
    min_years=1.0,      # you only have ~1.3y
    max_missing=0.05,
):
    metrics = pd.DataFrame({
        "missing_pct": returns.isnull().mean(),
        "years_data": len(returns) / 252,
    })

    passed = metrics[
        (metrics["years_data"] >= min_years) &
        (metrics["missing_pct"] <= max_missing)
    ].index.tolist()

    return passed, metrics.loc[passed]


# =====================================================
# Step 2: Regime sensitivity (HMM dispersion)
# =====================================================

def fit_market_hmm(
    market_returns: pd.Series,
    n_regimes: int = 2,
    random_state: int = 42,
):
    """Fit Gaussian HMM on market proxy returns."""
    
    hmm = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        random_state=random_state
    )
    hmm.fit(market_returns.values.reshape(-1, 1))
    regimes = hmm.predict(market_returns.values.reshape(-1, 1))
    return hmm, regimes


def regime_dispersion_scores(
    returns: pd.DataFrame,
    regimes: np.ndarray,
    market_index: pd.Index,
):
    """
    Compute regime dispersion score for each asset:
    std of regime-conditional Sharpe ratios.
    """
    
    dispersion = {}

    for asset in returns.columns:
        aligned = pd.DataFrame({
            "ret": returns[asset].reindex(market_index),
            "regime": regimes
        }).dropna()

        if aligned["regime"].nunique() < 2:
            dispersion[asset] = 0.0
            continue

        mu = aligned.groupby("regime")["ret"].mean()
        sigma = aligned.groupby("regime")["ret"].std()
        sr = mu / sigma
        dispersion[asset] = sr.std()

    return pd.Series(dispersion).sort_values(ascending=False)


# =====================================================
# Step 3: Diversification via clustering
# =====================================================

def cluster_assets(
    returns: pd.DataFrame,
    regime_mask: np.ndarray,
    max_clusters: int = 10,
):
    """Cluster assets using absolute correlation distance."""
    
    corr = returns.loc[regime_mask].corr()
    dist = 1 - np.abs(corr.values)
    np.fill_diagonal(dist, 0)

    Z = linkage(squareform(dist), method="ward")
    clusters = fcluster(Z, t=max_clusters, criterion="maxclust")

    return pd.Series(clusters, index=returns.columns)


def select_cluster_representatives(
    clusters: pd.Series,
    dispersion_scores: pd.Series,
    max_assets: int = 20,
):
    """Select best representative from each cluster."""
    
    selected = []

    for c in clusters.unique():
        members = clusters[clusters == c].index
        best = dispersion_scores.loc[members].idxmax()
        selected.append(best)

    return selected[:max_assets]


# =====================================================
# Step 4: Marginal portfolio contribution
# =====================================================

def marginal_contribution_scores(
    returns: pd.DataFrame,
    dispersion_scores: pd.Series,
):
    """Compute composite marginal contribution score."""
    
    scores = {}

    for asset in returns.columns:
        with_asset = returns
        without_asset = returns.drop(columns=[asset])

        # Sharpe contribution
        sr_with = annualized_sharpe(portfolio_returns(with_asset))
        sr_without = annualized_sharpe(portfolio_returns(without_asset))
        delta_sr = sr_with - sr_without

        # Drawdown contribution
        cum_with = (1 + portfolio_returns(with_asset)).cumprod()
        cum_without = (1 + portfolio_returns(without_asset)).cumprod()
        delta_dd = max_drawdown(cum_without) - max_drawdown(cum_with)

        score = (
            0.5 * delta_sr +
            0.3 * dispersion_scores.loc[asset] +
            0.2 * delta_dd
        )

        scores[asset] = score

    return pd.Series(scores).sort_values(ascending=False)


# =====================================================
# Main API
# =====================================================

def select_assets(
    candidate_returns: pd.DataFrame,
    market_proxy: str = "SPY",
    n_final: int = 15,
    n_regimes: int = 2,
):
    """
    Main entry point:
    returns list of selected assets.
    """
    # ---- Step 1: Data quality
    returns_q = candidate_returns.copy()

    # ---- Step 2: Regime sensitivity
    market_ret = returns_q[market_proxy].dropna()
    hmm, regimes = fit_market_hmm(market_ret, n_regimes=n_regimes)

    dispersion = regime_dispersion_scores(
        returns_q,
        regimes,
        market_ret.index
    )

    regime_assets = dispersion.head(30).index.tolist()
    returns_r = returns_q[regime_assets]

    # ---- Identify risk-on regime
    risk_on_regime = (
        market_ret.groupby(regimes).mean().idxmax()
    )
    risk_on_mask = regimes == risk_on_regime

    # ---- Step 3: Diversification
    clusters = cluster_assets(
        returns_r,
        risk_on_mask,
        max_clusters=10
    )

    diversified_assets = select_cluster_representatives(
        clusters,
        dispersion,
        max_assets=20
    )

    returns_d = returns_q[diversified_assets]

    # ---- Step 4: Portfolio contribution
    contribution = marginal_contribution_scores(
        returns_d,
        dispersion
    )

    final_assets = contribution.head(n_final).index.tolist()

    return final_assets, contribution, dispersion
