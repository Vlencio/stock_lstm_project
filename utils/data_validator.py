"""
Sanity checks for lookahead bias in training data.

These checks are heuristic, not mathematically conclusive. A correlation
above CORRELATION_WARNING_THRESHOLD between a feature and the target suggests
the feature may be leaking future information (e.g., computed incorrectly).
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Warn if any feature-to-target correlation exceeds this threshold.
CORRELATION_WARNING_THRESHOLD = 0.95


def validate_no_lookahead(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> None:
    """
    Checks for suspiciously high correlation between features and the target.

    Args:
        df: DataFrame containing features and target.
        feature_cols: List of feature column names to check.
        target_col: Name of the target column (e.g., next Close price).

    Raises:
        KeyError: If any specified column is not present in df.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: '{target_col}'")
    for col in feature_cols:
        if col not in df.columns:
            raise KeyError(f"Feature column not found: '{col}'")

    target = df[target_col].values

    for col in feature_cols:
        feat = df[col].values
        # Drop NaN positions before computing correlation
        mask = ~(np.isnan(feat) | np.isnan(target))
        if mask.sum() < 10:
            continue
        corr = float(np.corrcoef(feat[mask], target[mask])[0, 1])
        if abs(corr) >= CORRELATION_WARNING_THRESHOLD:
            logger.warning(
                "Lookahead bias risk: feature '%s' has correlation %.4f with target '%s'. "
                "Verify this indicator does not use future data.",
                col, corr, target_col,
            )

    logger.info("Data validation complete. Checked %d features against '%s'.", len(feature_cols), target_col)
