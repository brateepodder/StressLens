"""
shap_explanation.py  –  StressLens leading stress factor
=========================================================
Returns the single most influential physiological feature at the moment
stress was first detected for a given episode.

Public API
----------
    top_feature = get_leading_factor(episode, result_df, bundle)

    top_feature : dict with keys:
        name          : str    raw feature name  e.g. "BVP_PRV_RMSSD"
        display_name  : str    human-readable    e.g. "Heart Rate Variability (RMSSD)"
        shap_value    : float  positive = drives toward stress, negative = away
        feature_value : float  raw sensor value at trigger window
        direction     : str    "increases stress" | "decreases stress"
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

# ── Human-readable names for display in the questionnaire UI ─────────────────
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "BVP_PRV_RMSSD":    "Heart Rate Variability (RMSSD)",
    "BVP_PRV_pNN50":    "Heart Rate Variability (pNN50)",
    "BVP_PRV_SDNN":     "Heart Rate Variability (SDNN)",
    "BVP_HR_mean":      "Average Heart Rate",
    "BVP_HR_std":       "Heart Rate Variability",
    "BVP_PRV_IBI_mean": "Average Beat Interval",
    "BVP_PRV_IBI_std":  "Beat Interval Variability",
    "BVP_peakRate":     "Pulse Rate",
    "SCR_nPeaks":       "Skin Conductance Responses",
    "SCR_sumAmplitude": "Skin Conductance Response Strength",
    "SCR_area":         "Skin Conductance Response Area",
    "SCR_std":          "Skin Conductance Response Variability",
    "EDA_slope":        "Skin Conductance Trend",
    "EDA_mean":         "Average Skin Conductance",
    "EDA_std":          "Skin Conductance Variability",
    "SCL_mean":         "Baseline Skin Conductance",
    "SCL_std":          "Baseline Skin Conductance Variability",
    "SCL_timecorr":     "Skin Conductance Arousal Trend",
    "TEMP_slope":       "Skin Temperature Trend",
    "TEMP_mean":        "Average Skin Temperature",
    "TEMP_std":         "Skin Temperature Variability",
}


def get_leading_factor(
    episode: dict,
    result_df: pd.DataFrame,
    bundle: dict,
) -> dict | None:
    """
    Returns the single physiological feature most responsible for stress
    detection at the first stressed window of the given episode.

    Parameters
    ----------
    episode   : one episode dict from group_stress_episodes()
    result_df : per-window DataFrame with all features + predicted_stress
    bundle    : loaded model bundle dict from load_model()

    Returns
    -------
    dict with keys: name, display_name, shap_value, feature_value, direction
    None if SHAP is unavailable or no trigger window is found.
    """
    if not _SHAP_AVAILABLE:
        return None

    pipeline     = bundle["pipeline"]
    feature_cols = bundle["feature_cols"]

    # ── 1. Find the first stressed window of this episode ────────────────────
    trigger_mask = (
        (result_df["window_start_unix"] >= float(episode["start_unix"])) &
        (result_df["window_start_unix"] <  float(episode["end_unix"])) &
        (result_df["predicted_stress"]  == 2)
    )
    trigger_rows = result_df[trigger_mask]

    if trigger_rows.empty:
        return None

    trigger_row     = trigger_rows.sort_values("window_start_unix").iloc[0]
    window_features = trigger_row[feature_cols].to_numpy(dtype=float).copy()

    # ── 2. Build a small background from non-stressed windows ────────────────
    baseline_rows = result_df[result_df["predicted_stress"] != 2][feature_cols]
    if baseline_rows.empty:
        baseline_rows = result_df[feature_cols]          # fallback: use everything

    n_bg     = min(50, len(baseline_rows))
    bg_raw   = baseline_rows.sample(n=n_bg, random_state=42).to_numpy(dtype=float).copy()

    # Transform background through imputer + scaler (all steps except RF)
    bg_transformed = bg_raw.copy()
    for _, step in pipeline.steps[:-1]:
        bg_transformed = step.transform(bg_transformed)

    # ── 3. Run TreeExplainer on just the RF with transformed input ───────────
    rf_model = pipeline.steps[-1][1]

    explainer = shap.TreeExplainer(
        rf_model,
        data=bg_transformed,
        feature_perturbation="interventional",
        model_output="probability",
    )

    # Transform the single trigger window the same way
    window_transformed = window_features.reshape(1, -1).copy()
    for _, step in pipeline.steps[:-1]:
        window_transformed = step.transform(window_transformed)

    shap_vals = explainer.shap_values(window_transformed)

    # ── 4. Extract stress-class SHAP values ──────────────────────────────────
    classes    = list(rf_model.classes_)
    stress_idx = classes.index(2) if 2 in classes else 1

    if isinstance(shap_vals, list):
        sv = shap_vals[stress_idx][0]          # older SHAP: list of arrays per class
    else:
        sv = shap_vals[0, :, stress_idx]       # newer SHAP: (samples, features, classes)

    # ── 5. Return only the top feature ───────────────────────────────────────
    top_i         = int(np.argmax(np.abs(sv)))
    top_name      = feature_cols[top_i]
    top_shap      = float(sv[top_i])

    return {
        "name":          top_name,
        "display_name":  FEATURE_DISPLAY_NAMES.get(top_name, top_name),
        "shap_value":    top_shap,
        "feature_value": float(window_features[top_i]),
        "direction":     "increases stress" if top_shap > 0 else "decreases stress",
    }