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
from kalshi_core.clients.kalshi_ws import KalshiWsClient
from specialists.helpers import safe_int


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


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
        if val is not None and val >= 0:
            out[key] = int(val)
            return
        d = _safe_float(out.get(dollars_key))
        if d is not None and d >= 0:
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


def _market_from_ws_message(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if str(msg.get("type") or "").strip().lower() != "ticker":
        return None
    payload = msg.get("msg") if isinstance(msg.get("msg"), dict) else None
    if not isinstance(payload, dict):
        return None
    ticker = str(payload.get("market_ticker") or payload.get("ticker") or "").strip().upper()
    if not ticker:
        return None
    market = {
        "ticker": ticker,
        "yes_bid": safe_int(payload.get("yes_bid")),
        "yes_ask": safe_int(payload.get("yes_ask")),
        "no_bid": safe_int(payload.get("no_bid")),
        "no_ask": safe_int(payload.get("no_ask")),
        "status": str(payload.get("status") or "open").strip().lower() or "open",
        "event_ticker": str(payload.get("event_ticker") or "").strip().upper(),
        "title": str(payload.get("title") or ""),
    }
    return _normalize_market_quotes(market)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


class FeedState:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.market_quotes: Dict[str, Dict[str, Any]] = {}
        self.requested_tickers: Set[str] = set()
        self.ws_seen_tickers: Set[str] = set()
        self.last_ws_msg_epoch_s: float = 0.0
        self.last_publish_epoch_s: float = 0.0
        self.ws_generation: int = 0

    async def set_requested_tickers(self, tickers: Sequence[str]) -> None:
        requested = {str(t).strip().upper() for t in tickers if str(t).strip()}
        async with self._lock:
            self.requested_tickers = requested
            self.market_quotes = {k: v for k, v in self.market_quotes.items() if k in requested}
            self.ws_seen_tickers = {k for k in self.ws_seen_tickers if k in requested}

    async def get_requested_tickers(self) -> List[str]:
        async with self._lock:
            return sorted(self.requested_tickers)

    async def set_generation(self, generation: int) -> None:
        async with self._lock:
            self.ws_generation = int(generation)
            self.ws_seen_tickers = set()

    async def upsert_market(self, ticker: str, market: Dict[str, Any]) -> None:
        t = str(ticker or "").strip().upper()
        if not t:
            return
        async with self._lock:
            if not self.requested_tickers or t not in self.requested_tickers:
                return
            self.market_quotes[t] = dict(market)
            self.ws_seen_tickers.add(t)
            self.last_ws_msg_epoch_s = time.time()

    async def snapshot(self) -> Dict[str, Any]:
        async with self._lock:
            return {
                "market_quotes": {k: dict(v) for k, v in self.market_quotes.items()},
                "requested_tickers": sorted(self.requested_tickers),
                "ws_seen_tickers": sorted(self.ws_seen_tickers),
                "last_ws_msg_epoch_s": float(self.last_ws_msg_epoch_s),
                "last_publish_epoch_s": float(self.last_publish_epoch_s),
                "ws_generation": int(self.ws_generation),
            }

    async def mark_published(self) -> None:
        async with self._lock:
            self.last_publish_epoch_s = time.time()


async def _ticker_watch_task(*, feed_state: FeedState, state_paths: Sequence[Path], poll_seconds: float, stop_event: asyncio.Event) -> None:
    prev: Optional[List[str]] = None
    sleep_s = max(0.5, poll_seconds)
    while not stop_event.is_set():
        tickers = _tickers_from_states(state_paths)
        if prev != tickers:
            await feed_state.set_requested_tickers(tickers)
            print(f"[AB][FEED] watchlist updated requested={len(tickers)}")
            prev = list(tickers)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=sleep_s)
        except asyncio.TimeoutError:
            continue


async def _rest_backfill_task(
    *,
    feed_state: FeedState,
    base_url: str,
    poll_seconds: float,
    stop_event: asyncio.Event,
) -> None:
    loop = asyncio.get_running_loop()
    sleep_s = max(2.0, poll_seconds)
    with requests.Session() as session:
        while not stop_event.is_set():
            snap = await feed_state.snapshot()
            requested = snap["requested_tickers"]
            seen = set(snap["ws_seen_tickers"])
            missing = [t for t in requested if t not in seen]
            for ticker in missing:
                market = await loop.run_in_executor(
                    None,
                    lambda t=ticker: _fetch_market(session=session, base_url=base_url, ticker=t),
                )
                if market is not None:
                    await feed_state.upsert_market(ticker, market)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=sleep_s)
            except asyncio.TimeoutError:
                continue


async def _publish_task(*, feed_state: FeedState, feed_path: Path, poll_seconds: float, stop_event: asyncio.Event) -> None:
    interval_s = max(0.1, poll_seconds)
    while not stop_event.is_set():
        snap = await feed_state.snapshot()
        quotes = snap["market_quotes"]
        requested = snap["requested_tickers"]
        payload = {
            "updated_ts": _utc_now_iso(),
            "updated_epoch_s": time.time(),
            "tickers": quotes,
            "source": {
                "tickers_requested": len(requested),
                "tickers_published": len(quotes),
                "ws_quotes": len(snap["ws_seen_tickers"]),
                "rest_fallback": max(0, len(quotes) - len(snap["ws_seen_tickers"])),
                "ws_generation": snap["ws_generation"],
                "last_ws_msg_epoch_s": snap["last_ws_msg_epoch_s"],
            },
        }
        await asyncio.to_thread(_atomic_write_json, feed_path, payload)
        await feed_state.mark_published()
        print(
            f"[AB][FEED] publish requested={len(requested)} published={len(quotes)} "
            f"ws={len(snap['ws_seen_tickers'])} rest={max(0, len(quotes) - len(snap['ws_seen_tickers']))}"
        )
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            continue


async def _persistent_ws_task(
    *,
    feed_state: FeedState,
    use_private_auth: bool,
    stop_event: asyncio.Event,
    resubscribe_poll_s: float,
) -> None:
    client = KalshiWsClient(use_private_auth=use_private_auth)
    generation = 0
    backoff_s = 1.0
    subscribed_tickers: Set[str] = set()
    try:
        while not stop_event.is_set():
            generation += 1
            await feed_state.set_generation(generation)
            subscribed_tickers = set()
            try:
                print(
                    f"[AB][FEED] ws connect generation={generation} private_auth={int(use_private_auth)} url={client.ws_url}"
                )
                await client.connect()
                backoff_s = 1.0
                while not stop_event.is_set():
                    requested = set(await feed_state.get_requested_tickers())
                    to_subscribe = sorted(requested - subscribed_tickers)
                    if to_subscribe:
                        await client.subscribe(
                            channels=["ticker"],
                            market_tickers=to_subscribe,
                            send_initial_snapshot=True,
                        )
                        subscribed_tickers.update(to_subscribe)
                        print(
                            f"[AB][FEED] ws subscribe generation={generation} added={len(to_subscribe)} total={len(subscribed_tickers)}"
                        )

                    try:
                        msg = await asyncio.wait_for(client.recv_json(), timeout=max(1.0, resubscribe_poll_s))
                    except asyncio.TimeoutError:
                        continue

                    market = _market_from_ws_message(msg)
                    if market is None:
                        continue
                    ticker = str(market.get("ticker") or "").strip().upper()
                    if not ticker:
                        continue
                    await feed_state.upsert_market(ticker, market)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[AB][FEED] ws reconnect generation={generation} error={type(exc).__name__}: {exc}")
                try:
                    await client.close()
                except Exception:
                    pass
                if stop_event.is_set():
                    break
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=min(30.0, backoff_s))
                except asyncio.TimeoutError:
                    pass
                backoff_s = min(30.0, backoff_s * 2.0)
    finally:
        await client.close()


async def _async_main(args: argparse.Namespace) -> int:
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

    stop_event = asyncio.Event()
    feed_state = FeedState()
    tasks = [
        asyncio.create_task(
            _ticker_watch_task(
                feed_state=feed_state,
                state_paths=state_paths,
                poll_seconds=poll_seconds,
                stop_event=stop_event,
            ),
            name="shared-feed-watchlist",
        ),
        asyncio.create_task(
            _persistent_ws_task(
                feed_state=feed_state,
                use_private_auth=use_private_auth,
                stop_event=stop_event,
                resubscribe_poll_s=ws_timeout,
            ),
            name="shared-feed-ws",
        ),
        asyncio.create_task(
            _rest_backfill_task(
                feed_state=feed_state,
                base_url=base_url,
                poll_seconds=max(poll_seconds * 4.0, 2.0),
                stop_event=stop_event,
            ),
            name="shared-feed-rest",
        ),
        asyncio.create_task(
            _publish_task(
                feed_state=feed_state,
                feed_path=feed_path,
                poll_seconds=poll_seconds,
                stop_event=stop_event,
            ),
            name="shared-feed-publish",
        ),
    ]

    try:
        if max_runtime_minutes > 0:
            await asyncio.sleep(max_runtime_minutes * 60.0)
            stop_event.set()
        else:
            await asyncio.Future()
    except asyncio.CancelledError:
        stop_event.set()
        raise
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    print("[AB][FEED] stopped")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Shared Kalshi market-data feeder for A/B shadow runs.")
    parser.add_argument("--state-path", action="append", default=[], help="State JSON path to scan for runtime market tickers (repeatable).")
    parser.add_argument("--feed-path", required=True, help="Output JSON path for shared market feed.")
    parser.add_argument("--base-url", default=str(DEFAULT_BASE), help="Kalshi REST base URL.")
    parser.add_argument("--poll-seconds", type=float, default=0.5, help="Feed publish interval.")
    parser.add_argument("--ws-timeout-seconds", type=float, default=5.0, help="WS receive/resubscribe poll interval.")
    parser.add_argument("--max-runtime-minutes", type=float, default=0.0, help="Optional max runtime minutes; 0 means run forever.")
    args = parser.parse_args()
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
