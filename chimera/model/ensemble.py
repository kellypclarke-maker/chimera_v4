from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


def _clip_prob(p: float, eps: float = 1e-6) -> float:
    return max(float(eps), min(1.0 - float(eps), float(p)))


def logit(p: float) -> float:
    x = _clip_prob(float(p))
    return math.log(x / (1.0 - x))


def sigmoid(x: float) -> float:
    z = float(x)
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclass(frozen=True)
class EnsembleModel:
    intercept: float
    coef_logit_books: float
    coef_logit_moneypuck: float
    trained_on_rows: int
    notes: str = ""
    recency_half_life_days: float = 0.0
    recency_max_date: str = ""

    def predict_p_home(self, *, p_books: float, p_moneypuck: float) -> float:
        x = (
            float(self.intercept)
            + float(self.coef_logit_books) * logit(float(p_books))
            + float(self.coef_logit_moneypuck) * logit(float(p_moneypuck))
        )
        return sigmoid(x)

    def to_json_dict(self) -> dict:
        return {
            "intercept": float(self.intercept),
            "coef_logit_books": float(self.coef_logit_books),
            "coef_logit_moneypuck": float(self.coef_logit_moneypuck),
            "trained_on_rows": int(self.trained_on_rows),
            "notes": str(self.notes or ""),
            "recency_half_life_days": float(self.recency_half_life_days),
            "recency_max_date": str(self.recency_max_date or ""),
        }

    @staticmethod
    def from_json_dict(d: dict) -> "EnsembleModel":
        return EnsembleModel(
            intercept=float(d.get("intercept", 0.0)),
            coef_logit_books=float(d.get("coef_logit_books", 1.0)),
            coef_logit_moneypuck=float(d.get("coef_logit_moneypuck", 0.0)),
            trained_on_rows=int(d.get("trained_on_rows", 0)),
            notes=str(d.get("notes", "")),
            recency_half_life_days=float(d.get("recency_half_life_days", 0.0)),
            recency_max_date=str(d.get("recency_max_date", "")),
        )


def save_model(model: EnsembleModel, path: Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(model.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return p


def load_model(path: Path) -> EnsembleModel:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("invalid model file (expected JSON object)")
    return EnsembleModel.from_json_dict(data)


def fit_logit_ensemble(
    rows: Iterable[dict],
    *,
    aux_col: str = "moneypuck_p_home",
    aux_label: str = "moneypuck",
    require_books: bool = True,
    min_books_used: int = 1,
    recency_half_life_days: float = 0.0,
    recency_date_col: str = "date",
) -> Tuple[EnsembleModel, dict]:
    """
    Fit a logistic regression calibrator:
      logit(p_home) = a + b*logit(p_books) + c*logit(p_aux)
    """
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, log_loss
    except Exception as exc:
        raise RuntimeError("scikit-learn/numpy required to fit ensemble") from exc

    X = []
    y = []
    weights = []
    max_date: Optional[dt.date] = None
    if float(recency_half_life_days) > 0:
        for r in rows:
            s = str(r.get(recency_date_col) or "").strip()
            if not s:
                continue
            try:
                d = dt.date.fromisoformat(s)
            except Exception:
                continue
            if max_date is None or d > max_date:
                max_date = d
    for r in rows:
        try:
            y_raw = r.get("home_win")
            if y_raw in ("", None):
                continue
            yv = float(y_raw)
        except Exception:
            continue
        if yv not in (0.0, 1.0):
            continue

        # Books p_home (optional)
        pb = None
        try:
            v = r.get("books_p_home")
            if v not in ("", None):
                pb = float(v)
        except Exception:
            pb = None
        if require_books:
            try:
                bu = int(float(r.get("books_used") or 0))
            except Exception:
                bu = 0
            if pb is None or bu < int(min_books_used):
                continue

        # Auxiliary p_home (required): MoneyPuck for NHL, ESPN predictor for NBA/NFL, etc.
        pm = None
        try:
            v = r.get(str(aux_col))
            if v not in ("", None):
                pm = float(v)
        except Exception:
            pm = None
        if pm is None:
            continue

        if pb is None:
            pb = 0.5
        if not (0.0 <= pb <= 1.0 and 0.0 <= pm <= 1.0):
            continue

        weight: Optional[float] = None
        if float(recency_half_life_days) > 0:
            s = str(r.get(recency_date_col) or "").strip()
            if not s:
                continue
            try:
                d = dt.date.fromisoformat(s)
            except Exception:
                continue
            if max_date is None:
                continue
            age_days = max(0, (max_date - d).days)
            weight = 0.5 ** (float(age_days) / float(recency_half_life_days))

        X.append([logit(pb), logit(pm)])
        y.append(int(yv))
        if weight is not None:
            weights.append(float(weight))

    if not X:
        raise ValueError("no training rows after filtering")

    Xn = np.asarray(X, dtype=float)
    yn = np.asarray(y, dtype=int)

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        fit_intercept=True,
    )
    if float(recency_half_life_days) > 0 and len(weights) == len(y):
        clf.fit(Xn, yn, sample_weight=weights)
    else:
        clf.fit(Xn, yn)
    proba = clf.predict_proba(Xn)[:, 1]

    metrics = {
        "rows": int(len(y)),
        "brier": float(brier_score_loss(yn, proba)),
        "logloss": float(log_loss(yn, proba)),
        "intercept": float(clf.intercept_[0]),
        "coef_logit_books": float(clf.coef_[0][0]),
        "coef_logit_moneypuck": float(clf.coef_[0][1]),
        "aux_col": str(aux_col),
        "aux_label": str(aux_label),
        "recency_half_life_days": float(recency_half_life_days),
        "recency_max_date": "" if max_date is None else max_date.isoformat(),
    }

    model = EnsembleModel(
        intercept=metrics["intercept"],
        coef_logit_books=metrics["coef_logit_books"],
        coef_logit_moneypuck=metrics["coef_logit_moneypuck"],
        trained_on_rows=int(len(y)),
        notes=f"logit ensemble: a + b*logit(books) + c*logit({str(aux_label)})",
        recency_half_life_days=float(recency_half_life_days),
        recency_max_date="" if max_date is None else max_date.isoformat(),
    )
    return model, metrics

