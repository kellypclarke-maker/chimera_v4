#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kalshi_core.clients.kalshi_public import DEFAULT_MAIN_BASE as DEFAULT_BASE
from kalshi_core.clients.kalshi_ws import ws_collect_ticker_snapshot
from specialists.helpers import safe_int


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _parse_ts(raw: object) -> Optional[dt.datetime]:
    s = str(raw or "").strip()
    if not s:
        return None
    try:
        t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        return t
    except Exception:
        return None


def _safe_float(raw: object) -> Optional[float]:
    try:
        if raw is None:
            return None
        return float(raw)
    except Exception:
        return None


def _load_state(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _extract_runtime_ticker(order: Dict[str, Any]) -> str:
    rt = str(order.get("_runtime_market_ticker") or "").strip().upper()
    if rt:
        return rt
    return str(order.get("ticker") or "").strip().upper()


def _tickers_from_states(paths: Sequence[Path]) -> List[str]:
    out: Set[str] = set()
    for path in paths:
        state = _load_state(path)
        orders = state.get("orders")
        if not isinstance(orders, list):
            continue
        for order in orders:
            if not isinstance(order, dict):
                continue
            status = str(order.get("status") or "").strip().lower()
            if status not in {"open", "filled"}:
                continue
            ticker = _extract_runtime_ticker(order)
            if ticker:
                out.add(ticker)
    return sorted(out)


def _normalize_market_quotes(market: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(market)

    def _int_field(key: str, dollars_key: str) -> None:
        val = safe_int(out.get(key))
        if val is not None and val > 0:
            out[key] = int(val)
            return
        d = _safe_float(out.get(dollars_key))
        if d is not None and d > 0:
            out[key] = int(round(float(d) * 100.0))

    _int_field("yes_bid", "yes_bid_dollars")
    _int_field("yes_ask", "yes_ask_dollars")
    _int_field("no_bid", "no_bid_dollars")
    _int_field("no_ask", "no_ask_dollars")
    out["status"] = str(out.get("status") or "open").strip().lower() or "open"
    out["ticker"] = str(out.get("ticker") or "").strip().upper()
    out["event_ticker"] = str(out.get("event_ticker") or "").strip().upper()
    out["title"] = str(out.get("title") or "")
    return out


def _fetch_market(*, session: requests.Session, base_url: str, ticker: str) -> Optional[Dict[str, Any]]:
    url = f"{str(base_url).rstrip('/')}/markets/{ticker}"
    try:
        resp = session.get(url, timeout=20.0)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    market = payload.get("market") if isinstance(payload.get("market"), dict) else payload
    if not isinstance(market, dict):
        return None
    return _normalize_market_quotes(market)


def _ws_market_quotes(
    *,
    tickers: Sequence[str],
    use_private_auth: bool,
    timeout_s: float,
) -> Dict[str, Dict[str, Any]]:
    if not tickers:
        return {}
    try:
        snap = asyncio.run(
            ws_collect_ticker_snapshot(
                market_tickers=tickers,
                use_private_auth=bool(use_private_auth),
                timeout_s=max(0.5, float(timeout_s)),
            )
        )
    except Exception as exc:
        print(f"[AB][FEED] ws snapshot failed error={type(exc).__name__}: {exc}")
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for raw_ticker, msg in snap.items():
        if not isinstance(msg, dict):
            continue
        ticker = str(raw_ticker or "").strip().upper()
        if not ticker:
            continue
        out[ticker] = {
            "ticker": ticker,
            "yes_bid": safe_int(msg.get("yes_bid")),
            "yes_ask": safe_int(msg.get("yes_ask")),
            "no_bid": safe_int(msg.get("no_bid")),
            "no_ask": safe_int(msg.get("no_ask")),
            "status": "open",
            "event_ticker": str(msg.get("event_ticker") or "").strip().upper(),
            "title": str(msg.get("title") or ""),
        }
    return out


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _bool_env(name: str) -> bool:
    raw = str(os.environ.get(name) or "").strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Shared Kalshi market-data feeder for A/B shadow runs.")
    parser.add_argument("--state-path", action="append", default=[], help="State JSON path to scan for runtime market tickers (repeatable).")
    parser.add_argument("--feed-path", required=True, help="Output JSON path for shared market feed.")
    parser.add_argument("--base-url", default=str(DEFAULT_BASE), help="Kalshi REST base URL.")
    parser.add_argument("--poll-seconds", type=float, default=0.5, help="Feed loop poll interval.")
    parser.add_argument("--ws-timeout-seconds", type=float, default=5.0, help="WS snapshot timeout.")
    parser.add_argument("--max-runtime-minutes", type=float, default=0.0, help="Optional max runtime minutes; 0 means run forever.")
    args = parser.parse_args()

    state_paths = [Path(str(p)).resolve() for p in (args.state_path or []) if str(p).strip()]
    feed_path = Path(str(args.feed_path)).resolve()
    base_url = str(args.base_url or DEFAULT_BASE).strip().rstrip("/") or str(DEFAULT_BASE)
    poll_seconds = max(0.1, float(args.poll_seconds))
    ws_timeout = max(0.5, float(args.ws_timeout_seconds))
    max_runtime_minutes = max(0.0, float(args.max_runtime_minutes))

    key_id_present = bool(str(os.environ.get("KALSHI_API_KEY_ID") or "").strip())
    private_key_inline = bool(str(os.environ.get("KALSHI_API_PRIVATE_KEY") or "").strip())
    private_key_path = str(os.environ.get("KALSHI_PRIVATE_KEY_PATH") or "").strip()
    private_key_present = private_key_inline or (bool(private_key_path) and Path(private_key_path).exists())
    use_private_auth = key_id_present and private_key_present

    print(
        f"[AB][FEED] started feed_path={feed_path} states={len(state_paths)} poll_s={poll_seconds:.1f} "
        f"ws_timeout_s={ws_timeout:.1f} private_auth={use_private_auth}"
    )

    start_mono = time.monotonic()
    with requests.Session() as session:
        while True:
            now_mono = time.monotonic()
            if max_runtime_minutes > 0 and (now_mono - start_mono) >= (max_runtime_minutes * 60.0):
                break

            tickers = _tickers_from_states(state_paths)
            if not tickers:
                payload = {
                    "updated_ts": _utc_now_iso(),
                    "updated_epoch_s": time.time(),
                    "tickers": {},
                }
                _atomic_write_json(feed_path, payload)
                time.sleep(poll_seconds)
                continue

            ws_quotes = _ws_market_quotes(
                tickers=tickers,
                use_private_auth=use_private_auth,
                timeout_s=ws_timeout,
            )

            out_quotes: Dict[str, Dict[str, Any]] = dict(ws_quotes)
            missing = [t for t in tickers if t not in out_quotes]
            for ticker in missing:
                market = _fetch_market(session=session, base_url=base_url, ticker=ticker)
                if market is not None:
                    out_quotes[ticker] = market

            payload = {
                "updated_ts": _utc_now_iso(),
                "updated_epoch_s": time.time(),
                "tickers": out_quotes,
                "source": {
                    "tickers_requested": len(tickers),
                    "tickers_published": len(out_quotes),
                    "ws_quotes": len(ws_quotes),
                    "rest_fallback": max(0, len(out_quotes) - len(ws_quotes)),
                },
            }
            _atomic_write_json(feed_path, payload)
            print(
                f"[AB][FEED] publish requested={len(tickers)} published={len(out_quotes)} "
                f"ws={len(ws_quotes)} rest={max(0, len(out_quotes) - len(ws_quotes))}"
            )
            time.sleep(poll_seconds)

    print("[AB][FEED] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
