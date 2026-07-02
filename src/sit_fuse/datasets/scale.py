"""
Global min-max scaling utility. Kept as a standalone function, separate
from SFTemporalDataset, so the normalization strategy stays toggleable
rather than hardcoded -- per Nick's direction, this is the DEFAULT
strategy (vs. learnergy's built-in per-batch normalization), but still
being evaluated, not a permanent commitment.
"""
import numpy as np

# Mirrors learnergy.utils.constants.EPSILON's role: prevents division by
# zero when a feature's max equals its min (e.g. a constant channel).
EPSILON = 1e-10


def fit_minmax_scale(data: np.ndarray, feature_axis: int = -1):
    """
    Computes GLOBAL per-feature min/max statistics from data. Intended to
    be called ONCE on the TRAINING split only -- never on validation/test
    data, to avoid leaking information across the split (per the Data
    Note's "split by trajectory" requirement: stats must come only from
    trajectories assigned to train).

    :param data: ndarray to compute statistics from. Any number of
        dimensions; statistics are computed per-feature, where "feature"
        is whichever axis is `feature_axis`.
    :param feature_axis: axis that represents features (default: last
        axis, matching SFTemporalDataset's (n_timesteps, n_features) and
        (batch, seq_len, n_features) conventions).
    :return: (d_min, d_max), each an ndarray broadcastable against `data`.
    """
    axes = tuple(i for i in range(data.ndim) if i != feature_axis % data.ndim)
    d_min = data.min(axis=axes, keepdims=True)
    d_max = data.max(axis=axes, keepdims=True)
    return d_min, d_max


def apply_minmax_scale(data: np.ndarray, d_min: np.ndarray, d_max: np.ndarray):
    """
    Applies PRE-COMPUTED min/max statistics (from fit_minmax_scale) to
    scale data into [0, 1]. Use this for validation/test data, and for
    training data after fitting -- the same (d_min, d_max) must be reused
    everywhere, never recomputed per split.

    :param data: ndarray to scale.
    :param d_min: per-feature minimum, from fit_minmax_scale.
    :param d_max: per-feature maximum, from fit_minmax_scale.
    :return: scaled ndarray, same shape as `data`. Note: values from
        validation/test data that fall outside the training [min, max]
        range will scale to slightly below 0 or above 1 -- this is
        EXPECTED and not clipped, since clipping would hide genuinely
        out-of-distribution data rather than surfacing it.
    """
    return (data - d_min) / (d_max - d_min + EPSILON)


def minmax_scale(data: np.ndarray, feature_axis: int = -1):
    """
    Convenience wrapper: fit AND apply in one call, on the SAME data.
    Only appropriate when there is no train/val/test split to worry about
    (e.g. a single toy dataset, or exploratory analysis). For real
    train/val/test splits, use fit_minmax_scale on train, then
    apply_minmax_scale on all three splits with those same stats.

    :return: (scaled_data, d_min, d_max) -- stats are returned so they can
        still be reused later if needed (e.g. to inverse-transform).
    """
    d_min, d_max = fit_minmax_scale(data, feature_axis=feature_axis)
    scaled = apply_minmax_scale(data, d_min, d_max)
    return scaled, d_min, d_max
