import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


def fit_hmm_for_asset(
    features_df,
    asset,
    n_states=2,
    random_state=42,
    max_iter=1000
):
    """
    Fit a Gaussian HMM for a single asset using return and volatility features.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix containing columns like {ASSET}_ret, {ASSET}_vol
    asset : str
        Asset ticker (e.g. 'KO', 'PEP')
    n_states : int
        Number of hidden regimes
    random_state : int
        Random seed for reproducibility
    max_iter : int
        Maximum EM iterations

    Returns
    -------
    hmm_df : pd.DataFrame
        DataFrame with regime labels and probabilities
    model : GaussianHMM
        Fitted HMM model
    scaler : StandardScaler
        Scaler used for normalization
    """

    # --- 1. Select features ---
    cols = [f"{asset}_ret", f"{asset}_vol"]
    X = features_df[cols].dropna()

    # --- 2. Standardize ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 3. Fit HMM ---
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=max_iter,
        random_state=random_state
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
