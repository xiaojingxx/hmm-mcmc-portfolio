import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

def fit_hmm_for_asset(features_df, asset, n_states=2, random_state=42, max_iter=200, verbose=0):
    cols = [f"{asset}_ret", f"{asset}_vol"]
    X = features_df[cols].dropna()
    if len(X) < 50: 
        return pd.DataFrame(), None, None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=max_iter,
        random_state=random_state,
        tol=1e-3,
        verbose=verbose
    )
    
    model.fit(X_scaled)

    # 🔍 REGIME INSIGHTS (NEW)
    if verbose:
        print(f"  {asset}: Regimes fitted.")
        for i in range(n_states):
            ret_mean = model.means_[0, i]
            vol_mean = model.means_[1, i]
            print(f"    Regime {i}: ret={ret_mean:.4f}, vol={vol_mean:.4f} → {'HIGH-VOL' if vol_mean > 0 else 'LOW-VOL'}")

    states = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)

    hmm_df = X.copy()
    hmm_df["regime"] = states
    for i in range(n_states):
        hmm_df[f"prob_regime_{i}"] = probs[:, i]

    return hmm_df, model, scaler

def add_risk_scaling(hmm_df, asset, min_exposure=0.5, mom_window=5):
    vol_col = f"{asset}_vol"

    # Identify high-volatility regime (NEW: explicit check)
    regime_vols = hmm_df.groupby("regime")[vol_col].mean()
    high_vol_regime = regime_vols.idxmax()
    
    if regime_vols[0] > regime_vols[1]:
        print(f"  ⚠️ {asset}: Regime 0 ({regime_vols[0]:.4f}) > Regime 1 ({regime_vols[1]:.4f}) — FLIPPED!")
    
    print(f"  {asset}: High-vol regime = {high_vol_regime} (vol={regime_vols[high_vol_regime]:.4f})")
    
    hmm_df["p_high_vol"] = hmm_df[f"prob_regime_{high_vol_regime}"]
    hmm_df["risk_scale"] = min_exposure + (1 - min_exposure) * (1 - hmm_df["p_high_vol"])

    hmm_df["momentum"] = hmm_df[f"{asset}_ret"].rolling(mom_window).mean()
    hmm_df.loc[hmm_df["momentum"] > 0, "risk_scale"] = 1.0

    return hmm_df

def neg_sharpe(w, mu, Sigma):
    port_mu = w @ mu
    port_var = w @ Sigma @ w
    return -port_mu / np.sqrt(port_var) if port_var > 0 else 1e6

def optimize_regime_weights(mu, Sigma, max_gross=1.5):
    N = len(mu)
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': lambda w: max_gross - np.sum(np.abs(w))}
    ]
    bounds = [(-1, 1) for _ in range(N)]
    
    res = minimize(neg_sharpe, x0=np.ones(N)/N, args=(mu, Sigma),
                   method='SLSQP', bounds=bounds, constraints=constraints)
    return np.clip(res.x, -1, 1)  # Safety clip

def full_rolling_hmm_portfolio(features_df, asset_returns, window=252, step=21, verbose=True):
    """Enhanced with regime validation, summaries, and debug exports"""
    
    T = len(features_df)
    dates = features_df.index
    assets = [col.replace('_ret', '') for col in asset_returns.columns]
    
    weights_list, regime_list, date_list = [], [], []
    
    print(f"🚀 Rolling HMM Portfolio: T={T}, window={window}, step={step}")
    print(f"Assets: {assets}")
    
    for t in range(window, T, step):
        train_slice = slice(t - window, t)
        train_features = features_df.iloc[train_slice]
        train_rets = asset_returns.iloc[train_slice]
        
        # === HMM REGIMES ===
        all_regimes = []
        regime_vol_summary = {}
        
        for asset_ret_col in asset_returns.columns:
            asset = asset_ret_col.replace('_ret', '')
            try:
                hmm_df, _, _ = fit_hmm_for_asset(train_features, asset, verbose=verbose)
                hmm_df = add_risk_scaling(hmm_df, asset)
                all_regimes.append(hmm_df['regime'])
                regime_vol_summary[asset] = hmm_df.groupby('regime')[f'{asset}_vol'].mean()
            except Exception as e:
                print(f"  Skip {asset}: {e}")
                continue
        
        if len(all_regimes) == 0:
            continue
            
        # === STRICT: DROP NaN REGIME ROWS ===
        z_window_df = pd.concat(all_regimes, axis=1).mean(axis=1)
        valid_mask = ~z_window_df.isna()
        z_window = z_window_df[valid_mask].round().astype(int).values
        aligned_rets = train_rets.loc[z_window_df[valid_mask].index].values
        
        n_valid = len(z_window)
        print(f"t={t} ({dates[t].date()}): {len(train_rets)} raw → {n_valid} valid ({100*n_valid/len(train_rets):.0f}%)")
        
        if n_valid < 50:
            print(f"  SKIP: too few valid regimes")
            continue
            
        # === OPTIMIZE PER REGIME ===
        weights_reg = []
        for regime in [0, 1]:
            regime_mask = z_window == regime
            n_days = regime_mask.sum()
            if n_days > 20:
                regime_rets = aligned_rets[regime_mask]
                mu_reg = regime_rets.mean(axis=0)
                Sigma_reg = np.cov(regime_rets.T, bias=False)
                w_reg = optimize_regime_weights(mu_reg, Sigma_reg)
                print(f"  Regime {regime}: {n_days} days → SPY wt={w_reg[assets.index('SPY')]:.0%}")
            else:
                print(f"  Regime {regime}: only {n_days} days → equal weights")
                w_reg = np.ones(len(assets)) / len(assets)
            weights_reg.append(w_reg)
        
        # === SELECT CURRENT REGIME ===
        regime_t = int(z_window[-1])
        w_t = weights_reg[regime_t]
        
        # VALIDATION (NEW)
        spy_wt = w_t[assets.index('SPY')]
        expected = "LOW-VOL GROWTH (High SPY)" if regime_t == 0 else "HIGH-VOL PROTECT (Low SPY/Heavy TAIL)"
        print(f"  🎯 CURRENT: Regime {regime_t} | SPY={spy_wt:.0%} | {expected}")
        if regime_t == 1 and spy_wt > 0.45:
            print(f"  ⚠️ Regime 1 but SPY high → Check vol inputs!")
        
        assert np.abs(w_t).sum() <= 1.5 + 1e-6, "Gross exposure violated"
        
        weights_list.append(w_t)
        regime_list.append(regime_t)
        date_list.append(dates[t])
    
    # === FINAL SUMMARY ===
    weights_df = pd.DataFrame(weights_list, index=date_list, columns=assets)
    weights_df['regime'] = regime_list
    
    print(f"\n✅ Generated {len(weights_df)} periods!")
    print("\n📊 REGIME WEIGHTS SUMMARY:")
    print(weights_df.groupby('regime')[assets].mean().round(3))
    
    print("\n🔍 REGIME VOL CONFIRMATION (sample assets):")
    for asset in assets[:2]:
        try:
            _, _, _ = fit_hmm_for_asset(features_df, asset, verbose=False)
            vol_means = regime_vol_summary.get(asset, pd.Series([np.nan, np.nan], index=[0,1]))
            print(f"{asset}: Regime 0={vol_means[0]:.4f}, Regime 1={vol_means[1]:.4f}")
        except:
            pass
    
    # Export for debugging
    weights_df.to_csv('hmm_portfolio_weights_debug.csv')
    print("\n💾 Exported: hmm_portfolio_weights_debug.csv")
    
    return weights_df

# Usage example:
# weights = full_rolling_hmm_portfolio(features_df, asset_returns, verbose=True)
