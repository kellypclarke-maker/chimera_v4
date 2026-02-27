from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional


def _ceil_to_cent(dollars: float) -> float:
    return math.ceil(float(dollars) * 100.0) / 100.0


@dataclass(frozen=True, slots=True)
class FeeConfig:
    """
    Fee inputs are expressed in dollars per contract plus an optional
    Kalshi-style quadratic rate component.
    """

    maker_fee_flat: float = 0.0
    taker_fee_flat: float = 0.0
    maker_fee_rate: float = 0.0
    taker_fee_rate: float = 0.0

    @classmethod
    def from_env(cls, prefix: str = "KALSHI_") -> "FeeConfig":
        return cls(
            maker_fee_flat=_safe_float_env(f"{prefix}MAKER_FEE_FLAT", 0.0),
            taker_fee_flat=_safe_float_env(f"{prefix}TAKER_FEE_FLAT", 0.0),
            maker_fee_rate=_safe_float_env(f"{prefix}MAKER_FEE_RATE", 0.0),
            taker_fee_rate=_safe_float_env(f"{prefix}TAKER_FEE_RATE", 0.0),
        )


def _safe_float_env(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            return float(default)
        return float(raw)
    except Exception:
        return float(default)


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


def execution_fee_dollars(
    *,
    contracts: int,
    price: float,
    is_maker: bool = False,
    fee_config: Optional[FeeConfig] = None,
) -> float:
    cfg = fee_config or FeeConfig()
    n = max(0, int(contracts))
    if n <= 0:
        return 0.0
    flat = float(cfg.maker_fee_flat if is_maker else cfg.taker_fee_flat)
    if flat < 0.0:
        flat = 0.0
    rate_fee = (
        maker_fee_dollars(contracts=n, price=float(price), rate=float(cfg.maker_fee_rate))
        if is_maker
        else taker_fee_dollars(contracts=n, price=float(price), rate=float(cfg.taker_fee_rate))
    )
    return float(flat * float(n)) + float(rate_fee)


def expected_value_yes(
    *,
    p_true: float,
    price: float,
    fee: Optional[float] = None,
    adverse_selection_discount: float = 0.0,
    is_maker: bool = False,
    contracts: int = 1,
    fee_config: Optional[FeeConfig] = None,
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
    resolved_fee = (
        float(fee)
        if fee is not None
        else execution_fee_dollars(
            contracts=int(contracts),
            price=x,
            is_maker=bool(is_maker),
            fee_config=fee_config,
        )
    )
    return gross - float(resolved_fee)


def expected_value_no(
    *,
    p_true: float,
    price: float,
    fee: Optional[float] = None,
    adverse_selection_discount: float = 0.0,
    is_maker: bool = False,
    contracts: int = 1,
    fee_config: Optional[FeeConfig] = None,
) -> float:
    """
    Expected profit per 1 contract when buying NO at `price` (0..1 dollars).

    `p_true` should be the model probability of the NO side resolving true.
    """
    return expected_value_yes(
        p_true=float(p_true),
        price=float(price),
        fee=fee,
        adverse_selection_discount=float(adverse_selection_discount),
        is_maker=bool(is_maker),
        contracts=int(contracts),
        fee_config=fee_config,
    )
