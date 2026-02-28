from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple


def _normalize_binary_pair(p_a: float, p_b: float) -> Optional[Tuple[float, float]]:
    try:
        pa = float(p_a)
        pb = float(p_b)
    except Exception:
        return None
    total = pa + pb
    if total <= 0.0:
        return None
    pa /= total
    pb /= total
    if not (0.0 <= pa <= 1.0 and 0.0 <= pb <= 1.0):
        return None
    return (pa, pb)


def binary_shin_devig_from_decimal_odds(odds_a: float, odds_b: float) -> Optional[Tuple[float, float]]:
    """
    Binary Shin de-vigging.

    For two-outcome markets, Shin's method is analytically equivalent to the
    additive method. Given decimal odds o_a and o_b:

        q_i = 1 / o_i
        margin = (q_a + q_b) - 1
        p_i = q_i - margin / 2

    Reference equivalence: Clarke, Kovalchik, Ingram (2017), as cited by the
    `shin` package documentation.
    """
    try:
        oa = float(odds_a)
        ob = float(odds_b)
    except Exception:
        return None
    if oa <= 1.0 or ob <= 1.0:
        return None

    q_a = 1.0 / oa
    q_b = 1.0 / ob
    margin = (q_a + q_b) - 1.0
    p_a = q_a - (margin / 2.0)
    p_b = q_b - (margin / 2.0)

    # In realistic two-way sports books these should stay positive. If they do
    # not, fall back to multiplicative normalization instead of emitting an
    # impossible probability pair.
    if p_a <= 0.0 or p_b <= 0.0:
        return _normalize_binary_pair(q_a, q_b)
    return _normalize_binary_pair(p_a, p_b)


def binary_shin_devig_named_odds(odds_by_team: Mapping[str, float]) -> Optional[Dict[str, float]]:
    if len(odds_by_team) != 2:
        return None
    items = list(odds_by_team.items())
    team_a, odds_a = items[0]
    team_b, odds_b = items[1]
    probs = binary_shin_devig_from_decimal_odds(float(odds_a), float(odds_b))
    if probs is None:
        return None
    p_a, p_b = probs
    return {str(team_a): float(p_a), str(team_b): float(p_b)}
