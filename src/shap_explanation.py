"""
shap_explanation.py  –  StressLens per-episode SHAP explanations
=================================================================
Generates SHAP feature-importance explanations for each detected stress
episode, using the FIRST stressed window of the episode as the trigger point.

Public API
----------
    explanations = explain_episodes(episodes, result_df, bundle)

    explanations : list[dict]  –  one dict per episode, each containing:
        episode_idx     : int
        start_iso       : str   (copied from episode)
        shap_values     : np.ndarray  shape (n_features,)
        feature_names   : list[str]
        feature_values  : np.ndarray  shape (n_features,)  (raw, unscaled)
        base_value      : float       (expected model output = baseline stress prob)
        stress_prob     : float       (model's stress probability at trigger window)
        top_features    : list[dict]  top-N features sorted by |SHAP value|
                          each dict: {name, shap_value, feature_value, direction}
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# SHAP is an optional dependency — give a clear error if missing
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


# Number of background samples to use for the SHAP masker.
# More = more accurate but slower. 50–100 is a good balance for real-time use.
N_BACKGROUND_SAMPLES = 50

# Number of top features to return per episode
TOP_N_FEATURES = 5


def _check_shap():
    if not _SHAP_AVAILABLE:
        raise ImportError(
            "SHAP is not installed. Run: pip install shap\n"
            "Then restart your Streamlit app."
        )


def _get_pipeline_input(pipeline, X_raw: np.ndarray) -> np.ndarray:
    """
    Runs X_raw through all pipeline steps EXCEPT the final estimator.
    This gives us the transformed feature matrix that the RandomForest actually sees.
    Used to build the background dataset in the transformed space.
    """
    X_transformed = X_raw.copy()
    steps = pipeline.steps  # list of (name, estimator) tuples

    # Apply all steps except the last (the classifier)
    for name, step in steps[:-1]:
        X_transformed = step.transform(X_transformed)

    return X_transformed


def build_background(
    result_df: pd.DataFrame,
    feature_cols: list[str],
    pipeline,
    n_samples: int = N_BACKGROUND_SAMPLES,
    label_baseline: int = 1,
) -> np.ndarray:
    """
    Builds a background dataset for SHAP from non-stressed windows.

    SHAP uses this background to answer: "compared to a typical non-stressed
    moment, how much does each feature push this window toward stress?"

    We use non-stressed windows because they represent the physiological
    baseline — the "normal" state we're measuring deviation from.

    Parameters
    ----------
    result_df     : full per-window DataFrame with predicted_stress column
    feature_cols  : ordered feature column list from the bundle
    pipeline      : the fitted sklearn Pipeline from the bundle
    n_samples     : how many background rows to sample (more = slower but better)
    label_baseline: integer label for non-stressed class (1 in your training)

    Returns
    -------
    background : np.ndarray shape (n_samples, n_features)
                 in the TRANSFORMED space (after imputer + scaler)
    """
    _check_shap()

    # Get non-stressed windows
    baseline_mask = result_df["predicted_stress"] == label_baseline
    baseline_df   = result_df[baseline_mask][feature_cols]

    if len(baseline_df) == 0:
        # Edge case: everything classified as stressed — use all windows
        baseline_df = result_df[feature_cols]

    # Sample up to n_samples rows
    n = min(n_samples, len(baseline_df))
    sampled = baseline_df.sample(n=n, random_state=42).to_numpy(dtype=float).copy()

    # Transform through imputer + scaler (all steps except the final classifier)
    # This puts the background in the same space the RandomForest was trained on
    background_transformed = _get_pipeline_input(pipeline, sampled)

    return background_transformed


def explain_episode_window(
    window_features: np.ndarray,
    pipeline,
    background: np.ndarray,
    feature_cols: list[str],
    threshold: float,
    label_stress: int = 2,
) -> dict:
    """
    Runs SHAP on a single window (the first stressed window of an episode).

    Uses shap.TreeExplainer on just the RandomForest component of the pipeline,
    with pre-transformed input (so the scaler/imputer have already been applied).
    This is ~10x faster than wrapping the full pipeline with shap.Explainer
    and gives identical results.

    Parameters
    ----------
    window_features : np.ndarray shape (n_features,)  RAW (unscaled) feature values
    pipeline        : the full fitted sklearn Pipeline
    background      : np.ndarray shape (n_bg, n_features) TRANSFORMED background
    feature_cols    : ordered list of feature names
    threshold       : decision threshold from bundle
    label_stress    : integer label for stress class (2)

    Returns
    -------
    dict with shap_values, feature_names, feature_values, base_value, stress_prob
    """
    _check_shap()

    # Transform the single window through imputer + scaler
    window_raw         = window_features.reshape(1, -1).copy()
    window_transformed = _get_pipeline_input(pipeline, window_raw)

    # Get the RandomForest (final step of pipeline)
    rf_model = pipeline.steps[-1][1]  # last step's estimator

    # Build a TreeExplainer on the RF using the transformed background
    # TreeExplainer natively understands RandomForest decision paths —
    # much faster and more exact than the generic KernelExplainer
    explainer = shap.TreeExplainer(
        rf_model,
        data=background,
        feature_perturbation="interventional",  # correct for correlated features
        model_output="probability",             # output in probability space, not log-odds
    )

    # Compute SHAP values for the single transformed window
    # shap_values has shape (1, n_features, n_classes)
    shap_vals = explainer.shap_values(window_transformed)

    # Find which class index corresponds to stress label
    classes    = list(rf_model.classes_)
    stress_idx = classes.index(label_stress) if label_stress in classes else 1

    # Extract stress class SHAP values for this window → shape (n_features,)
    if isinstance(shap_vals, list):
        # Older SHAP versions return a list of arrays, one per class
        sv = shap_vals[stress_idx][0]
    else:
        # Newer SHAP versions return a single array (n_samples, n_features, n_classes)
        sv = shap_vals[0, :, stress_idx]

    # Base value = expected stress probability across the background dataset
    base_value = float(explainer.expected_value[stress_idx]) \
        if hasattr(explainer.expected_value, "__len__") \
        else float(explainer.expected_value)

    # Raw stress probability for this window
    stress_probs = pipeline.predict_proba(window_raw)
    stress_prob  = float(stress_probs[0, stress_idx])

    # Build top-N feature list sorted by absolute SHAP value
    abs_sv    = np.abs(sv)
    top_idx   = np.argsort(abs_sv)[::-1][:TOP_N_FEATURES]

    top_features = []
    for i in top_idx:
        top_features.append({
            "name":          feature_cols[i],
            "shap_value":    float(sv[i]),
            "feature_value": float(window_features[i]),   # raw unscaled value for display
            "direction":     "increases stress" if sv[i] > 0 else "decreases stress",
        })

    return {
        "shap_values":    sv,
        "feature_names":  feature_cols,
        "feature_values": window_features,
        "base_value":     base_value,
        "stress_prob":    stress_prob,
        "top_features":   top_features,
    }


def explain_episodes(
    episodes: list[dict],
    result_df: pd.DataFrame,
    bundle: dict,
) -> list[dict]:
    """
    Generates SHAP explanations for each detected stress episode.

    For each episode, we take the FIRST stressed window — the moment the model
    first detected stress onset — as the trigger point for explanation. This
    tells us: "what physiological features caused the model to first flag stress?"

    Parameters
    ----------
    episodes  : list of episode dicts from group_stress_episodes()
    result_df : per-window DataFrame with all features + predicted_stress
    bundle    : loaded model bundle dict from load_model()

    Returns
    -------
    list of explanation dicts, one per episode, each containing:
        episode_idx, start_iso, end_iso, duration_sec  (from episode)
        shap_values, feature_names, feature_values     (full SHAP output)
        base_value, stress_prob                        (model internals)
        top_features                                   (top-N for display)
    """
    _check_shap()

    if not episodes:
        return []

    pipeline     = bundle["pipeline"]
    feature_cols = bundle["feature_cols"]
    threshold    = bundle["threshold"]

    # Build the background dataset once — reused for all episodes
    background = build_background(result_df, feature_cols, pipeline)

    explanations = []

    for ep_idx, episode in enumerate(episodes):
        # ── Find the first stressed window inside this episode ────────────────
        # The episode's start_unix is the window_start_unix of the first
        # stressed window, so we match on that exactly.
        ep_start_unix = float(episode["start_unix"])

        trigger_mask = (
            (result_df["window_start_unix"] >= ep_start_unix) &
            (result_df["window_start_unix"] <  float(episode["end_unix"])) &
            (result_df["predicted_stress"]  == 2)
        )
        trigger_rows = result_df[trigger_mask]

        if trigger_rows.empty:
            # Shouldn't happen, but skip gracefully if episode windows not found
            continue

        # Take the chronologically first stressed window
        trigger_row = trigger_rows.sort_values("window_start_unix").iloc[0]

        # Extract raw (unscaled) feature vector for this window
        window_features = trigger_row[feature_cols].to_numpy(dtype=float).copy()

        # ── Run SHAP ──────────────────────────────────────────────────────────
        try:
            shap_result = explain_episode_window(
                window_features=window_features,
                pipeline=pipeline,
                background=background,
                feature_cols=feature_cols,
                threshold=threshold,
            )
        except Exception as e:
            # Don't crash the whole app if SHAP fails on one episode
            shap_result = {
                "shap_values":    np.zeros(len(feature_cols)),
                "feature_names":  feature_cols,
                "feature_values": window_features,
                "base_value":     0.5,
                "stress_prob":    float(trigger_row.get("stress_prob", 0.0)),
                "top_features":   [],
                "error":          str(e),
            }

        explanations.append({
            # Episode metadata
            "episode_idx":  ep_idx,
            "start_iso":    episode["start_iso"],
            "end_iso":      episode["end_iso"],
            "duration_sec": episode["duration_sec"],
            # SHAP results
            **shap_result,
        })

    return explanations


def format_explanation_for_display(explanation: dict) -> str:
    """
    Formats a single episode explanation as a readable text summary.
    Useful for populating the stress questionnaire UI alongside the form.
    """
    lines = [
        f"**Stress onset detected at:** {explanation['start_iso']}",
        f"**Model stress probability:** {explanation['stress_prob']:.1%}",
        f"**Baseline (expected) probability:** {explanation['base_value']:.1%}",
        "",
        f"**Top physiological triggers (SHAP analysis):**",
    ]

    for rank, feat in enumerate(explanation["top_features"], 1):
        arrow  = "⬆️" if feat["shap_value"] > 0 else "⬇️"
        impact = abs(feat["shap_value"])
        lines.append(
            f"  {rank}. {arrow} **{feat['name']}** "
            f"(value: {feat['feature_value']:.4f}, "
            f"SHAP impact: {feat['shap_value']:+.4f}) "
            f"— {feat['direction']}"
        )

    return "\n".join(lines)