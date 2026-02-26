from __future__ import annotations

import math


def _ceil_to_cent(dollars: float) -> float:
    return math.ceil(float(dollars) * 100.0) / 100.0


def taker_fee_dollars(*, contracts: int, price: float, rate: float = 0.07) -> float:
    """
    Kalshi taker fee model (quadratic) used across the v2c tools:

      fee = ceil_to_cent(rate * contracts * P * (1-P))

    where P is the executed contract price in probability units (0..1).
    """
    n = int(contracts)
    if n <= 0:
        return 0.0
    p = float(price)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(_ceil_to_cent(float(rate) * float(n) * p * (1.0 - p)))


def maker_fee_dollars(*, contracts: int, price: float, rate: float = 0.0) -> float:
    """
    Kalshi maker fee model (same functional form as taker in these tools):

      fee = ceil_to_cent(rate * contracts * P * (1-P))

    Use `rate=0.0` to ignore maker fees, or set to your current maker fee rate.
    """
    n = int(contracts)
    if n <= 0:
        return 0.0
    p = float(price)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    r = float(rate)
    if r <= 0.0:
        return 0.0
    return float(_ceil_to_cent(r * float(n) * p * (1.0 - p)))


def expected_value_yes(
    *,
    p_true: float,
    price: float,
    fee: float = 0.0,
    adverse_selection_discount: float = 0.0,
) -> float:
    """
    Expected profit per 1 contract when buying YES at `price` (0..1 dollars).

    Payout: + (1 - price) if event happens else -price.
    """
    p = float(p_true)
    x = float(price)
    if not (0.0 <= p <= 1.0) or not (0.0 < x < 1.0):
        return float("nan")
    conditional_p_true = max(0.0, p * (1.0 - float(adverse_selection_discount)))
    gross = conditional_p_true * (1.0 - x) + (1.0 - conditional_p_true) * (-x)
    return gross - float(fee)
