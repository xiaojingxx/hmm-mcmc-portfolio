import numpy as np
import pandas as pd
from scipy.optimize import minimize

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


def fit_hmm_for_asset(features_df, asset, n_states=2, random_state=42, max_iter=200, verbose=0):
    cols = [f"{asset}_ret", f"{asset}_vol"]
    X = features_df[cols].dropna()
    if len(X) < 50: return pd.DataFrame(), None, None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=max_iter,
        random_state=random_state,
        tol=1e-3,          # Faster convergence
        verbose=verbose    # 0 = silent
    )
    
    model.fit(X_scaled)

    # --- 4. Infer regimes ---
    states = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)

    # --- 5. Output dataframe ---
    hmm_df = X.copy()
    hmm_df["regime"] = states

    for i in range(n_states):
        hmm_df[f"prob_regime_{i}"] = probs[:, i]

    return hmm_df, model, scaler


def add_risk_scaling(hmm_df, asset, min_exposure=0.5, mom_window=5):
    """
    Add regime-based risk scaling using high-volatility state + momentum filter.
    """
    vol_col = f"{asset}_vol"

    # Identify high-volatility regime
    high_vol_regime = hmm_df.groupby("regime")[vol_col].mean().idxmax()
    hmm_df["p_high_vol"] = hmm_df[f"prob_regime_{high_vol_regime}"]

    # Base risk scaling (capped)
    hmm_df["risk_scale"] = min_exposure + (1 - min_exposure) * (1 - hmm_df["p_high_vol"])

    # Momentum filter: override scaling if momentum positive
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
    return res.x

def full_rolling_hmm_portfolio(features_df, asset_returns, window=252, step=21):
    """NO fillna(0) - DROP NaN regimes only"""
    
    T = len(features_df)
    dates = features_df.index
    assets = [col.replace('_ret', '') for col in asset_returns.columns]
    
    weights_list, regime_list, date_list = [], [], []
    
    print(f"🚀 Rolling (drop NaN regimes): T={T}, window={window}")
    
    for t in range(window, T, step):
        train_slice = slice(t - window, t)
        train_features = features_df.iloc[train_slice]
        train_rets = asset_returns.iloc[train_slice]
        
        # === HMM REGIMES (NO FILLNA) ===
        all_regimes = []
        for asset_ret_col in asset_returns.columns:
            asset = asset_ret_col.replace('_ret', '')
            try:
                hmm_df, _, _ = fit_hmm_for_asset(train_features, asset)
                hmm_df = add_risk_scaling(hmm_df, asset)
                # NO fillna(0)! Keep NaN regimes as-is
                all_regimes.append(hmm_df['regime'])
            except Exception as e:
                print(f"  Skip {asset}: {e}")
                continue
        
        if len(all_regimes) == 0:
            continue
            
        # === STRICT: DROP NaN REGIME ROWS ===
        z_window_df = pd.concat(all_regimes, axis=1).mean(axis=1)
        
        # DROP rows where ANY regime is NaN
        valid_mask = ~z_window_df.isna()
        z_window = z_window_df[valid_mask].round().astype(int).values
        aligned_rets = train_rets.loc[z_window_df[valid_mask].index].values
        
        print(f"t={t}: {len(train_rets)} raw → {len(z_window)} valid regimes ({100*len(z_window)/len(train_rets):.0f}%)")
        
        if len(z_window) < 50:
            print(f"  SKIP: too few valid regimes ({len(z_window)})")
            continue
            
        # === OPTIMIZATION (only valid regimes) ===
        weights_reg = []
        for regime in [0, 1]:
            regime_mask = z_window == regime
            if regime_mask.sum() > 20:
                regime_rets = aligned_rets[regime_mask]
                mu_reg = regime_rets.mean(axis=0)
                Sigma_reg = np.cov(regime_rets.T)
                w_reg = optimize_regime_weights(mu_reg, Sigma_reg)
                print(f"  Regime {regime}: {regime_mask.sum()} days ✓")
            else:
                print(f"  Regime {regime}: only {regime_mask.sum()} days → equal weights")
                w_reg = np.ones(len(assets)) / len(assets)
            weights_reg.append(w_reg)
        
        regime_t = int(z_window[-1])
        w_t = weights_reg[regime_t]
        
        weights_list.append(w_t)
        regime_list.append(regime_t)
        date_list.append(dates[t])
    
    # === RESULTS ===
    weights_df = pd.DataFrame(weights_list, index=date_list, columns=assets)
    weights_df['regime'] = regime_list
    
    print(f"\n✅ {len(weights_df)} valid periods generated!")
    print("Regime breakdown:")
    print(weights_df['regime'].value_counts())
    
    return weights_df