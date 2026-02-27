#!/usr/bin/env python3
import argparse
import asyncio
import csv
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from chimera.clients.kalshi import fetch_event, fetch_market
from chimera.fees import expected_value_yes
from chimera.trading.autotrade import ExecutionEngine, ManagedOrder
from chimera.trading.crypto_oracle_v4 import CryptoOracle
from chimera.trading.weather_oracle_v4 import WeatherOracle


def _safe_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _kalshi_public_base() -> str:
    return (
        str(os.environ.get("KALSHI_API_BASE") or "")
        or str(os.environ.get("KALSHI_PUBLIC_BASE") or "")
        or str(os.environ.get("KALSHI_BASE") or "")
        or "https://api.elections.kalshi.com/trade-api/v2"
    ).strip().rstrip("/")


def log_shadow_trade(
    oracle_type: str,
    ticker: str,
    action: str,
    price_cents: int,
    quantity: int,
    spot_price: Optional[float],
    expected_value: Optional[float],
) -> None:
    ledger_path = os.path.join(os.path.dirname(__file__), "shadow_ledger.csv")
    headers = [
        "timestamp",
        "oracle_type",
        "ticker",
        "action",
        "price_cents",
        "quantity",
        "spot_price",
        "expected_value",
    ]
    write_header = not os.path.exists(ledger_path)
    timestamp = datetime.now(timezone.utc).isoformat()

    with open(ledger_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)
        writer.writerow(
            [
                timestamp,
                str(oracle_type),
                str(ticker),
                str(action),
                int(price_cents),
                int(quantity),
                "" if spot_price is None else float(spot_price),
                "" if expected_value is None else float(expected_value),
            ]
        )


class ShadowExecutionEngine(ExecutionEngine):
    def __init__(self, tickers, edge_threshold_cents):
        super().__init__(tickers, edge_threshold_cents)
        self._latest_spot_by_ticker: Dict[str, Optional[float]] = {}
        self._oracle_type_by_ticker: Dict[str, str] = {}

        # Override client methods for shadow mode to strictly enforce paper trading.
        engine = self

        class MockClient:
            def __init__(self, engine_ref: "ShadowExecutionEngine"):
                self.engine = engine_ref

            def cancel_order(self, order_id):
                print(f"[SHADOW] Cancelled order {order_id}")

            def place_order(self, ticker, side, action, count, price_cents):
                order_id = f"shadow-{int(time.time())}"
                print(
                    f"[SHADOW] {action.upper()} {side.upper()} order placed: {order_id} "
                    f"for {ticker} at {price_cents}c x {count}"
                )
                self.engine._post_shadow_order_log(
                    ticker=str(ticker),
                    side=str(side),
                    action=str(action),
                    count=int(count),
                    price_cents=int(price_cents),
                )
                return {"order_id": order_id, "order": {"order_id": order_id}}

        self.client = MockClient(engine)

    def _oracle_type_for_ticker(self, ticker: str) -> str:
        t = str(ticker).strip().upper()
        oracle_type = str(self._oracle_type_by_ticker.get(t) or "").strip().lower()
        if oracle_type:
            return oracle_type
        if t.startswith("KXBTC-") or "BTC" in t:
            return "crypto"
        return "weather"

    def _post_shadow_order_log(
        self,
        *,
        ticker: str,
        side: str,
        action: str,
        count: int,
        price_cents: int,
    ) -> None:
        if str(action).strip().lower() != "buy" or str(side).strip().lower() != "yes":
            return
        t = str(ticker).strip().upper()
        p_true = self.dynamic_p_true.get(t)
        ev: Optional[float] = None
        if p_true is not None:
            try:
                ev = expected_value_yes(
                    p_true=float(p_true),
                    price=float(int(price_cents)) / 100.0,
                    fee=0.0,
                    adverse_selection_discount=0.05,
                )
            except Exception:
                ev = None
        log_shadow_trade(
            oracle_type=self._oracle_type_for_ticker(t),
            ticker=t,
            action="buy_yes",
            price_cents=int(price_cents),
            quantity=int(count),
            spot_price=self._latest_spot_by_ticker.get(t),
            expected_value=ev,
        )


class FullShadowEngine(ShadowExecutionEngine):
    def __init__(
        self,
        tickers,
        edge_threshold_cents,
        *,
        size_contracts: int,
        crypto_refresh_seconds: float,
        target_date: str,
    ):
        super().__init__(tickers, edge_threshold_cents)
        for mo in self.managed_orders.values():
            mo.remaining_count = int(size_contracts)

        self._size_contracts = int(size_contracts)
        self._crypto_refresh_seconds = float(crypto_refresh_seconds)
        self._last_crypto_refresh_ts = 0.0
        self._cached_crypto_tickers: List[str] = []
        self._target_date = str(target_date).strip().upper()
        if not self._target_date:
            raise ValueError("target_date is required")
        self._kalshi_session = requests.Session()

    def _ensure_ticker_state(self, ticker: str) -> None:
        t = str(ticker).strip().upper()
        if not t:
            return
        if t not in self.tickers:
            self.tickers.append(t)
        if t not in self.kalshi_order_book:
            self.kalshi_order_book[t] = {"yes_bid": None, "yes_ask": None}
        if t not in self.managed_orders:
            self.managed_orders[t] = ManagedOrder(ticker=t, remaining_count=self._size_contracts)
        if t not in self.filled_contracts:
            self.filled_contracts[t] = 0

    @staticmethod
    def _extract_btc_strike(ticker: str) -> Optional[float]:
        t = str(ticker).strip().upper()
        if not t.startswith("KXBTC-"):
            return None
        try:
            suffix = t.split("-")[-1]
            if not suffix.startswith("T"):
                return None
            return float(int(suffix[1:]))
        except Exception:
            return None

    def _fetch_btc_spot_rest_fallback(self) -> Optional[float]:
        urls = (
            "https://api.binance.us/api/v3/ticker/price?symbol=BTCUSDT",
            "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
        )
        for url in urls:
            try:
                r = self._kalshi_session.get(url, timeout=5.0)
                r.raise_for_status()
                data = r.json() if r.content else {}
                px = float(data.get("price"))
                if px > 0.0:
                    return px
            except Exception:
                continue
        return None

    async def _resolve_spot(self, crypto_oracle: CryptoOracle) -> Optional[float]:
        if crypto_oracle.spot_price is not None and float(crypto_oracle.spot_price) > 0.0:
            return float(crypto_oracle.spot_price)
        return await asyncio.to_thread(self._fetch_btc_spot_rest_fallback)

    @staticmethod
    def _market_strike_bounds(market: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        strike_type = str(market.get("strike_type") or "").strip().lower()
        floor = _safe_float(market.get("floor_strike"))
        cap = _safe_float(market.get("cap_strike"))

        if strike_type == "greater":
            return floor, None
        if strike_type == "less":
            return None, cap
        return floor, cap

    @classmethod
    def _market_strike_hint(cls, market: Dict[str, Any]) -> Optional[float]:
        lo, hi = cls._market_strike_bounds(market)
        if lo is not None and hi is not None:
            return (lo + hi) / 2.0
        return lo if lo is not None else hi

    @classmethod
    def _market_in_window(
        cls,
        market: Dict[str, Any],
        lower_bound: float,
        upper_bound: float,
    ) -> bool:
        lo, hi = cls._market_strike_bounds(market)
        if lo is not None and hi is not None:
            return not (hi < lower_bound or lo > upper_bound)
        if lo is not None:
            return lo <= upper_bound
        if hi is not None:
            return hi >= lower_bound
        return False

    def _fetch_open_btc_events_for_date(self) -> List[str]:
        base = _kalshi_public_base()
        prefix = f"KXBTC-{self._target_date}"
        out: List[str] = []
        cursor = ""

        for _ in range(10):
            params: Dict[str, Any] = {"series_ticker": "KXBTC", "status": "open", "limit": 200}
            if cursor:
                params["cursor"] = cursor
            r = self._kalshi_session.get(f"{base}/events", params=params, timeout=10.0)
            r.raise_for_status()
            payload = r.json() if r.content else {}
            if not isinstance(payload, dict):
                break

            events = payload.get("events")
            if not isinstance(events, list):
                break

            for event in events:
                if not isinstance(event, dict):
                    continue
                event_ticker = str(event.get("event_ticker") or "").strip().upper()
                if not event_ticker:
                    continue
                if event_ticker.startswith(prefix):
                    out.append(event_ticker)

            next_cursor = str(payload.get("cursor") or "").strip()
            if not next_cursor:
                break
            if next_cursor == cursor:
                break
            cursor = next_cursor

        return sorted(set(out))

    async def _discover_viable_crypto_tickers(
        self,
        current_spot: float,
        variance_pct: float = 0.02,
    ) -> List[str]:
        lower_bound = float(current_spot) * (1.0 - float(variance_pct))
        upper_bound = float(current_spot) * (1.0 + float(variance_pct))

        try:
            event_tickers = await asyncio.to_thread(self._fetch_open_btc_events_for_date)
        except Exception as exc:
            print(f"[SHADOW] Failed to list open BTC events for {self._target_date}: {exc}")
            return []

        if not event_tickers:
            print(f"[SHADOW] No open KXBTC events found for {self._target_date}")
            return []

        candidates: Dict[str, Optional[float]] = {}
        for event_ticker in event_tickers:
            try:
                event = await asyncio.to_thread(
                    fetch_event,
                    event_ticker=event_ticker,
                    session=self._kalshi_session,
                )
            except Exception as exc:
                print(f"[SHADOW] Failed to fetch event {event_ticker}: {exc}")
                continue

            markets = event.get("markets") if isinstance(event, dict) else None
            if not isinstance(markets, list):
                continue

            for market in markets:
                if not isinstance(market, dict):
                    continue
                status = str(market.get("status") or "").strip().lower()
                if status and status not in {"open", "active"}:
                    continue
                ticker = str(market.get("ticker") or "").strip().upper()
                if not ticker:
                    continue
                if not self._market_in_window(market, lower_bound, upper_bound):
                    continue
                candidates[ticker] = self._market_strike_hint(market)

        def _sort_key(ticker: str) -> Tuple[int, float, str]:
            hint = candidates.get(ticker)
            if hint is None:
                return (1, 0.0, ticker)
            return (0, abs(float(hint) - float(current_spot)), ticker)

        tickers = sorted(candidates.keys(), key=_sort_key)
        print(
            f"[SHADOW] Discovered viable BTC markets from {len(event_tickers)} active event(s): "
            f"{', '.join(tickers) if tickers else '<none>'}"
        )
        return tickers

    async def _refresh_viable_crypto_tickers(self, current_spot: float) -> List[str]:
        now = time.time()
        should_refresh = (
            self._last_crypto_refresh_ts <= 0.0
            or (now - self._last_crypto_refresh_ts) >= self._crypto_refresh_seconds
        )
        if should_refresh:
            self._cached_crypto_tickers = await self._discover_viable_crypto_tickers(
                float(current_spot),
                variance_pct=0.02,
            )
            self._last_crypto_refresh_ts = now
            print(
                f"[SHADOW] Refreshed viable BTC ticker set ({len(self._cached_crypto_tickers)}): "
                f"{', '.join(self._cached_crypto_tickers) if self._cached_crypto_tickers else '<none>'}"
            )
        return list(self._cached_crypto_tickers)

    async def _fetch_kalshi_book(self, ticker: str) -> Tuple[Optional[int], Optional[int]]:
        try:
            market: Dict[str, Any] = await asyncio.to_thread(
                fetch_market,
                ticker=str(ticker).strip().upper(),
                session=self._kalshi_session,
            )
        except Exception as exc:
            print(f"[SHADOW] Kalshi fetch failed for {ticker}: {exc}")
            return None, None

        yes_bid = _safe_int(market.get("yes_bid"))
        yes_ask = _safe_int(market.get("yes_ask"))
        return yes_bid, yes_ask

    async def _process_crypto_cycle(self, crypto_oracle: CryptoOracle) -> None:
        spot = await self._resolve_spot(crypto_oracle)
        if spot is None:
            print("[SHADOW] BTC spot unavailable from WS and REST fallback; skipping crypto cycle")
            return

        crypto_tickers = await self._refresh_viable_crypto_tickers(float(spot))
        if not crypto_tickers:
            print("[SHADOW] No viable BTC tickers discovered; skipping crypto cycle")
            return

        for ticker in crypto_tickers:
            self._ensure_ticker_state(ticker)
            yes_bid, yes_ask = await self._fetch_kalshi_book(ticker)
            self._oracle_type_by_ticker[str(ticker).strip().upper()] = "crypto"
            self._latest_spot_by_ticker[str(ticker).strip().upper()] = float(spot)

            # EV Calculation Hook placeholder requested by spec.
            print(f"[SHADOW] Checking EV for {ticker} | Spot: {spot:.2f} | Kalshi Ask: {yes_ask}")

            strike = self._extract_btc_strike(ticker)
            if strike is None:
                dynamic_p = 0.5
            else:
                # Reuse existing CryptoOracle probability mapping per strike.
                strike_oracle = CryptoOracle("btcusdt", kalshi_strike_price=float(strike), flash_threshold=50.0)
                strike_oracle.spot_price = float(spot)
                strike_oracle.mass_cancel_flag = bool(crypto_oracle.mass_cancel_flag)
                dynamic_p = strike_oracle.calculate_p_true(time_to_expiry_hours=24.0, volatility=0.5)

            async with self.lock:
                self.dynamic_p_true[ticker] = float(dynamic_p)
                book = self.kalshi_order_book[ticker]
                book["yes_bid"] = yes_bid
                book["yes_ask"] = yes_ask

            if crypto_oracle.mass_cancel_flag:
                await self._mass_cancel(ticker)
            else:
                await self._evaluate_order(ticker)

    async def _process_weather_cycle(self) -> None:
        weather_tickers = [
            f"KXHIGHNY-{self._target_date}-B40.5",
            f"KXHIGHNY-{self._target_date}-B42.5",
        ]
        for wx_t in weather_tickers:
            self._ensure_ticker_state(wx_t)
            self._oracle_type_by_ticker[str(wx_t).strip().upper()] = "weather"
            self._latest_spot_by_ticker[str(wx_t).strip().upper()] = None
            async with self.lock:
                self.dynamic_p_true[wx_t] = 0.5
            await self._evaluate_order(wx_t)

    async def oracle_listener(self):
        print("Starting Night Watch Shadow Oracles (Crypto, Weather)...")

        # Primary live spot feed from Binance websocket.
        crypto_oracle = CryptoOracle("btcusdt", kalshi_strike_price=68000.0, flash_threshold=50.0)
        weather_oracle = WeatherOracle("NYC")

        asyncio.create_task(crypto_oracle.listen_binance_ws())
        asyncio.create_task(weather_oracle.poll_noaa_forecast())

        while True:
            await asyncio.sleep(self.poll_seconds)
            await self._process_crypto_cycle(crypto_oracle)
            await self._process_weather_cycle()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-contracts", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=1.5)
    parser.add_argument("--crypto-refresh-seconds", type=float, default=300.0)
    parser.add_argument("--target-date", required=True, help="Kalshi date token, e.g. 26FEB26")
    args = parser.parse_args()

    target_date = str(args.target_date).strip().upper()
    tickers = [
        f"KXHIGHNY-{target_date}-B40.5",
        f"KXHIGHNY-{target_date}-B42.5",
    ]

    engine = FullShadowEngine(
        tickers=tickers,
        edge_threshold_cents=5.0,
        size_contracts=int(args.size_contracts),
        crypto_refresh_seconds=float(args.crypto_refresh_seconds),
        target_date=target_date,
    )
    engine.poll_seconds = float(args.poll_seconds)
    await engine.run()


if __name__ == "__main__":
    asyncio.run(main())
