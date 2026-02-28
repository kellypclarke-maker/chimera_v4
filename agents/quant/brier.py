from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd


def _coerce_outcome(raw: Any) -> float:
    if raw is None:
        return float("nan")
    if isinstance(raw, (bool, np.bool_)):
        return 1.0 if bool(raw) else 0.0
    if isinstance(raw, (int, float, np.integer, np.floating)):
        v = float(raw)
        if v in {0.0, 1.0}:
            return v
        return float("nan")
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "win", "won"}:
        return 1.0
    if s in {"0", "false", "no", "loss", "lost"}:
        return 0.0
    return float("nan")


def _coerce_probability(raw: Any) -> float:
    try:
        v = float(raw)
    except Exception:
        return float("nan")
    if not np.isfinite(v):
        return float("nan")
    if 0.0 <= v <= 1.0:
        return v
    return float("nan")


def _prepare_binary_arrays(
    y_true: Sequence[Any],
    y_prob: Sequence[Any],
) -> Tuple[np.ndarray, np.ndarray]:
    yt = np.asarray([_coerce_outcome(v) for v in y_true], dtype=float)
    yp = np.asarray([_coerce_probability(v) for v in y_prob], dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask]


def compute_brier_score(y_true: Sequence[Any], y_prob: Sequence[Any]) -> float:
    """
    Standard Brier score for binary outcomes.

    Lower is better. Perfect calibration/perfect discrimination yields 0.0.
    """
    yt, yp = _prepare_binary_arrays(y_true, y_prob)
    if yt.size == 0:
        return float("nan")
    return float(np.mean((yp - yt) ** 2))


def expected_calibration_error(
    y_true: Sequence[Any],
    y_prob: Sequence[Any],
    n_bins: int = 10,
) -> float:
    yt, yp = _prepare_binary_arrays(y_true, y_prob)
    if yt.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, max(2, int(n_bins)) + 1)
    ece = 0.0
    total = float(yt.size)
    for idx in range(len(bins) - 1):
        lo = bins[idx]
        hi = bins[idx + 1]
        if idx == len(bins) - 2:
            mask = (yp >= lo) & (yp <= hi)
        else:
            mask = (yp >= lo) & (yp < hi)
        if not np.any(mask):
            continue
        bin_prob = float(np.mean(yp[mask]))
        bin_true = float(np.mean(yt[mask]))
        ece += abs(bin_prob - bin_true) * (float(np.sum(mask)) / total)
    return float(ece)


def reliability_curve(
    y_true: Sequence[Any],
    y_prob: Sequence[Any],
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    yt, yp = _prepare_binary_arrays(y_true, y_prob)
    if yt.size == 0:
        return []
    bins = np.linspace(0.0, 1.0, max(2, int(n_bins)) + 1)
    out: List[Dict[str, float]] = []
    for idx in range(len(bins) - 1):
        lo = float(bins[idx])
        hi = float(bins[idx + 1])
        if idx == len(bins) - 2:
            mask = (yp >= lo) & (yp <= hi)
        else:
            mask = (yp >= lo) & (yp < hi)
        count = int(np.sum(mask))
        if count == 0:
            out.append(
                {
                    "bin_index": float(idx),
                    "bin_lower": lo,
                    "bin_upper": hi,
                    "count": 0.0,
                    "predicted_mean": float("nan"),
                    "observed_mean": float("nan"),
                    "gap": float("nan"),
                }
            )
            continue
        pred = float(np.mean(yp[mask]))
        obs = float(np.mean(yt[mask]))
        out.append(
            {
                "bin_index": float(idx),
                "bin_lower": lo,
                "bin_upper": hi,
                "count": float(count),
                "predicted_mean": pred,
                "observed_mean": obs,
                "gap": float(pred - obs),
            }
        )
    return out


def brier_full_report(
    ledger_df: pd.DataFrame,
    *,
    p_true_col: str = "p_true",
    outcome_col: str = "outcome",
    n_bins: int = 10,
) -> Dict[str, Any]:
    if not isinstance(ledger_df, pd.DataFrame):
        raise TypeError("ledger_df must be a pandas DataFrame")
    if p_true_col not in ledger_df.columns:
        raise KeyError(f"Missing probability column: {p_true_col}")
    if outcome_col not in ledger_df.columns:
        raise KeyError(f"Missing outcome column: {outcome_col}")

    y_true = ledger_df[outcome_col].tolist()
    y_prob = ledger_df[p_true_col].tolist()
    yt, yp = _prepare_binary_arrays(y_true, y_prob)

    return {
        "brier_score": compute_brier_score(yt, yp),
        "ece": expected_calibration_error(yt, yp, n_bins=n_bins),
        "sample_size": int(yt.size),
        "reliability_bins": reliability_curve(yt, yp, n_bins=n_bins),
    }
