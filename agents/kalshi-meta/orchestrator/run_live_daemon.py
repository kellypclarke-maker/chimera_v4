#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import json
import re
import math
import os
import requests
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable or "python"


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _utc_day() -> str:
    return _utc_now().date().isoformat()


def _run_cmd(cmd: List[str]) -> Tuple[bool, int, str, str]:
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    ok = proc.returncode == 0
    return (ok, proc.returncode, proc.stdout or "", proc.stderr or "")


def _parse_ts(x: object) -> Optional[dt.datetime]:
    s = str(x or "").strip()
    if not s:
        return None
    try:
        t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        return t
    except Exception:
        return None


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _safe_float(x: object) -> Optional[float]:
    try:
        y = float(str(x or "").strip())
    except Exception:
        return None
    if not math.isfinite(y):
        return None
    return float(y)


def _candidate_action_ok(row: Dict[str, str]) -> bool:
    return str(row.get("action") or "").strip().lower() == "post_yes" and str(row.get("maker_or_taker") or "").strip().lower() == "maker"


def _load_watch_tickers(
    *,
    day_key: str,
    watch_top_n: int,
    min_ev_dollars: float,
    max_close_hours: float,
) -> List[str]:
    now = _utc_now()
    latest_close = now + dt.timedelta(hours=max(0.0, float(max_close_hours)))
    out: List[str] = []
    seen: set[str] = set()

    # Prioritize actively selected live tickets.
    tickets_csv = ROOT / "reports" / "orders" / f"{day_key}_live_order_tickets.csv"
    for row in _read_csv_rows(tickets_csv):
        t = str(row.get("ticker") or "").strip().upper()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)

    candidates_csv = ROOT / "reports" / "daily" / f"{day_key}_candidates.csv"
    rows = _read_csv_rows(candidates_csv)
    best_ev_by_ticker: Dict[str, float] = {}
    for row in rows:
        t = str(row.get("ticker") or "").strip().upper()
        if not t:
            continue
        if not _candidate_action_ok(row):
            continue
        ev = _safe_float(row.get("ev_dollars"))
        if ev is None or ev < float(min_ev_dollars):
            continue
        close_t = _parse_ts(row.get("close_time"))
        if close_t is None or close_t < now or close_t > latest_close:
            continue
        prev = best_ev_by_ticker.get(t)
        if prev is None or ev > prev:
            best_ev_by_ticker[t] = ev

    for t, _ in sorted(best_ev_by_ticker.items(), key=lambda kv: kv[1], reverse=True):
        if t in seen:
            continue
        seen.add(t)
        out.append(t)

    cap = max(1, int(watch_top_n))
    return out[:cap]


def _coerce_price_cents(x: object) -> Optional[float]:
    v = _safe_float(x)
    if v is None:
        return None
    if v < 0.0 or v > 100.0:
        return None
    return float(v)


def _extract_yes_reference_price(msg: Dict[str, Any]) -> Optional[float]:
    bid_keys = ("yes_bid", "yesBid", "best_yes_bid", "yes_bid_price", "yesBidPrice")
    ask_keys = ("yes_ask", "yesAsk", "best_yes_ask", "yes_ask_price", "yesAskPrice")
    mid_keys = ("yes_mid", "yesMid", "yes_price", "yesPrice", "price", "last_price", "lastPrice")

    bid: Optional[float] = None
    ask: Optional[float] = None
    for k in bid_keys:
        if k in msg:
            bid = _coerce_price_cents(msg.get(k))
            if bid is not None:
                break
    for k in ask_keys:
        if k in msg:
            ask = _coerce_price_cents(msg.get(k))
            if ask is not None:
                break
    if bid is not None and ask is not None and ask >= bid:
        return float((bid + ask) / 2.0)

    for k in mid_keys:
        if k in msg:
            px = _coerce_price_cents(msg.get(k))
            if px is not None:
                return px
    return None


POLYMARKET_SYMBOL_BY_EVENT_PREFIX: Dict[str, str] = {
    "KXBTC15M": "btcusdt",
    "KXBTCD": "btcusdt",
    "KXBTC": "btcusdt",
    "KXETH15M": "ethusdt",
    "KXETHD": "ethusdt",
    "KXETH": "ethusdt",
    "KXXRPD": "xrpusdt",
    "KXXRP": "xrpusdt",
    "KXSOLE": "solusdt",
    "KXSOLD": "solusdt",
    "KXSOL": "solusdt",
}


def _parse_csv_values(raw: str) -> List[str]:
    return [str(x).strip() for x in str(raw or "").split(",") if str(x).strip()]


def _polymarket_symbol_for_kalshi_ticker(ticker: str) -> Optional[str]:
    tok = str(ticker or "").strip().upper()
    if not tok:
        return None
    for prefix in sorted(POLYMARKET_SYMBOL_BY_EVENT_PREFIX.keys(), key=len, reverse=True):
        if tok.startswith(prefix):
            return str(POLYMARKET_SYMBOL_BY_EVENT_PREFIX[prefix])
    return None


def _infer_polymarket_symbol_map_from_tickers(market_tickers: Sequence[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for ticker in market_tickers:
        tok = str(ticker or "").strip().upper()
        if not tok:
            continue
        symbol = _polymarket_symbol_for_kalshi_ticker(tok)
        if not symbol:
            continue
        out.setdefault(symbol, [])
        if tok not in out[symbol]:
            out[symbol].append(tok)
    return out


def _infer_polymarket_symbols_from_tickers(market_tickers: Sequence[str]) -> List[str]:
    return sorted(_infer_polymarket_symbol_map_from_tickers(market_tickers).keys())


def _resolve_polymarket_symbols(
    *,
    market_tickers: Sequence[str],
    manual_csv: str,
    allow_manual_unaligned: bool,
) -> Tuple[List[str], Dict[str, List[str]]]:
    aligned_map = _infer_polymarket_symbol_map_from_tickers(market_tickers)
    manual = [s.lower() for s in _parse_csv_values(manual_csv)]
    if not manual:
        return (sorted(aligned_map.keys()), aligned_map)

    out: List[str] = []
    seen: set[str] = set()
    for s in manual:
        if (not bool(allow_manual_unaligned)) and (s not in aligned_map):
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        aligned_map.setdefault(s, [])
    return (out, aligned_map)


def _extract_polymarket_symbol_price(msg: Dict[str, Any]) -> Optional[Tuple[str, float]]:
    payload = msg.get("payload") if isinstance(msg.get("payload"), dict) else msg
    if not isinstance(payload, dict):
        return None

    symbol_raw = payload.get("symbol") or payload.get("asset") or payload.get("ticker")
    symbol = str(symbol_raw or "").strip().lower()
    if not symbol:
        return None

    for k in ("value", "price", "last", "mid", "mark"):
        px = _safe_float(payload.get(k))
        if px is not None and px > 0.0:
            return (symbol, float(px))
    return None


def _public_base() -> str:
    base = (
        str(os.environ.get("KALSHI_PUBLIC_BASE") or "")
        or str(os.environ.get("KALSHI_BASE") or "")
        or str(os.environ.get("KALSHI_API_BASE") or "")
        or "https://api.elections.kalshi.com/trade-api/v2"
    ).strip()
    return base.rstrip("/")


async def _await_polymarket_move_trigger(
    *,
    symbols: Sequence[str],
    symbol_to_kalshi_tickers: Dict[str, List[str]],
    timeout_s: float,
    min_move_pct: float,
    ws_url: str,
) -> Dict[str, Any]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from kalshi_core.clients.polymarket_ws import PolymarketRtdsClient

    watch = [str(s).strip().lower() for s in symbols if str(s).strip()]
    if not watch:
        return {"triggered": False, "reason": "no_symbols", "watch_count": 0}

    want = set(watch)
    baseline: Dict[str, float] = {}
    client = PolymarketRtdsClient(ws_url=ws_url if str(ws_url).strip() else None)
    await client.connect()
    try:
        await client.subscribe_crypto_prices(symbols=watch)
        deadline = time.time() + max(0.1, float(timeout_s))
        while time.time() < deadline:
            rem = max(0.1, deadline - time.time())
            try:
                ws_msg = await asyncio.wait_for(client.recv_json(), timeout=rem)
            except asyncio.TimeoutError:
                break

            parsed = _extract_polymarket_symbol_price(ws_msg)
            if parsed is None:
                continue
            symbol, px = parsed
            if symbol not in want:
                continue

            base = baseline.get(symbol)
            if base is None:
                baseline[symbol] = float(px)
                continue

            move_abs = abs(float(px) - float(base))
            move_pct = (move_abs / max(1e-9, abs(float(base)))) * 100.0
            if float(min_move_pct) > 0.0 and move_pct >= float(min_move_pct):
                return {
                    "triggered": True,
                    "reason": "polymarket_price_move",
                    "symbol": symbol,
                    "matched_kalshi_tickers": list(symbol_to_kalshi_tickers.get(symbol) or []),
                    "baseline_price": round(float(base), 8),
                    "latest_price": round(float(px), 8),
                    "move_abs": round(float(move_abs), 8),
                    "move_pct": round(float(move_pct), 4),
                    "threshold_pct": float(min_move_pct),
                    "watch_count": len(watch),
                }
    finally:
        await client.close()

    return {
        "triggered": False,
        "reason": "timeout",
        "watch_count": len(watch),
        "baseline_ready": len(baseline),
    }


async def _await_ws_move_trigger(
    *,
    market_tickers: Sequence[str],
    timeout_s: float,
    min_move_pct: float,
    min_move_cents: float,
    use_private_auth: bool,
) -> Dict[str, Any]:
    # Import lazily so non-WS usage has no runtime dependency on websockets.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from kalshi_core.clients.kalshi_ws import KalshiWsClient

    watch = [str(t).strip().upper() for t in market_tickers if str(t).strip()]
    if not watch:
        return {"triggered": False, "reason": "no_tickers"}

    want = set(watch)
    baseline: Dict[str, float] = {}
    latest: Dict[str, float] = {}

    client = KalshiWsClient(use_private_auth=bool(use_private_auth))
    await client.connect()
    try:
        await client.subscribe(channels=["ticker"], market_tickers=watch, send_initial_snapshot=True)
        deadline = time.time() + max(0.1, float(timeout_s))
        while time.time() < deadline:
            rem = max(0.1, deadline - time.time())
            try:
                ws_msg = await asyncio.wait_for(client.recv_json(), timeout=rem)
            except asyncio.TimeoutError:
                break

            if str(ws_msg.get("type") or "").strip().lower() != "ticker":
                continue
            payload = ws_msg.get("msg") if isinstance(ws_msg.get("msg"), dict) else {}
            ticker = str(payload.get("market_ticker") or "").strip().upper()
            if ticker not in want:
                continue
            px = _extract_yes_reference_price(payload)
            if px is None:
                continue

            latest[ticker] = float(px)
            base = baseline.get(ticker)
            if base is None:
                baseline[ticker] = float(px)
                continue

            move_cents = abs(float(px) - float(base))
            move_pct = (move_cents / max(1.0, abs(float(base)))) * 100.0
            hit_cents = float(min_move_cents) > 0.0 and move_cents >= float(min_move_cents)
            hit_pct = float(min_move_pct) > 0.0 and move_pct >= float(min_move_pct)
            if hit_cents or hit_pct:
                return {
                    "triggered": True,
                    "reason": "price_move",
                    "ticker": ticker,
                    "baseline_yes_price_cents": round(float(base), 4),
                    "latest_yes_price_cents": round(float(px), 4),
                    "move_cents": round(float(move_cents), 4),
                    "move_pct": round(float(move_pct), 4),
                    "threshold_pct": float(min_move_pct),
                    "threshold_cents": float(min_move_cents),
                    "watch_count": len(watch),
                }
    finally:
        await client.close()

    return {
        "triggered": False,
        "reason": "timeout",
        "watch_count": len(watch),
        "baseline_ready": len(baseline),
    }


async def _await_any_market_trigger(
    *,
    kalshi_watch: Sequence[str],
    poly_symbols: Sequence[str],
    poly_symbol_to_kalshi_tickers: Dict[str, List[str]],
    timeout_s: float,
    kalshi_min_move_pct: float,
    kalshi_min_move_cents: float,
    kalshi_use_private_auth: bool,
    polymarket_enabled: bool,
    polymarket_min_move_pct: float,
    polymarket_ws_url: str,
) -> Dict[str, Any]:
    tasks: Dict[str, asyncio.Task[Dict[str, Any]]] = {}
    tasks["kalshi"] = asyncio.create_task(
        _await_ws_move_trigger(
            market_tickers=kalshi_watch,
            timeout_s=float(timeout_s),
            min_move_pct=float(kalshi_min_move_pct),
            min_move_cents=float(kalshi_min_move_cents),
            use_private_auth=bool(kalshi_use_private_auth),
        )
    )
    if bool(polymarket_enabled) and poly_symbols:
        tasks["polymarket"] = asyncio.create_task(
            _await_polymarket_move_trigger(
                symbols=poly_symbols,
                symbol_to_kalshi_tickers=poly_symbol_to_kalshi_tickers,
                timeout_s=float(timeout_s),
                min_move_pct=float(polymarket_min_move_pct),
                ws_url=str(polymarket_ws_url or ""),
            )
        )

    outcomes: Dict[str, Dict[str, Any]] = {}
    while tasks:
        done, _ = await asyncio.wait(tasks.values(), return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            source = next((k for k, v in tasks.items() if v is t), "")
            if not source:
                continue
            del tasks[source]
            try:
                out = t.result()
            except Exception as exc:
                out = {"triggered": False, "reason": "ws_error", "error": type(exc).__name__}

            out.setdefault("source", source)
            outcomes[source] = out
            if bool(out.get("triggered")):
                for pend in list(tasks.values()):
                    pend.cancel()
                if tasks:
                    await asyncio.gather(*tasks.values(), return_exceptions=True)
                return out

    kalshi_out = outcomes.get("kalshi") or {"triggered": False, "reason": "timeout"}
    poly_out = outcomes.get("polymarket")
    return {
        "triggered": False,
        "reason": str(kalshi_out.get("reason") or "timeout"),
        "watch_count": int(kalshi_out.get("watch_count") or len(kalshi_watch)),
        "kalshi": kalshi_out,
        "polymarket": poly_out or {},
    }


def _poll_move_trigger(
    *,
    market_tickers: Sequence[str],
    timeout_s: float,
    min_move_pct: float,
    min_move_cents: float,
    poll_interval_s: float,
) -> Dict[str, Any]:
    watch = [str(t).strip().upper() for t in market_tickers if str(t).strip()]
    if not watch:
        return {"triggered": False, "reason": "no_tickers", "watch_count": 0}

    session = requests.Session()
    base = _public_base()
    baseline: Dict[str, float] = {}
    deadline = time.time() + max(0.1, float(timeout_s))

    while time.time() < deadline:
        for ticker in watch:
            try:
                r = session.get(f"{base}/markets/{ticker}", timeout=15.0)
                r.raise_for_status()
                payload = r.json()
                market = payload.get("market") if isinstance(payload, dict) and isinstance(payload.get("market"), dict) else payload
                if not isinstance(market, dict):
                    continue
            except Exception:
                continue

            px = _extract_yes_reference_price(market)
            if px is None:
                continue

            base_px = baseline.get(ticker)
            if base_px is None:
                baseline[ticker] = float(px)
                continue

            move_cents = abs(float(px) - float(base_px))
            move_pct = (move_cents / max(1.0, abs(float(base_px)))) * 100.0
            hit_cents = float(min_move_cents) > 0.0 and move_cents >= float(min_move_cents)
            hit_pct = float(min_move_pct) > 0.0 and move_pct >= float(min_move_pct)
            if hit_cents or hit_pct:
                return {
                    "triggered": True,
                    "reason": "poll_price_move",
                    "ticker": ticker,
                    "baseline_yes_price_cents": round(float(base_px), 4),
                    "latest_yes_price_cents": round(float(px), 4),
                    "move_cents": round(float(move_cents), 4),
                    "move_pct": round(float(move_pct), 4),
                    "threshold_pct": float(min_move_pct),
                    "threshold_cents": float(min_move_cents),
                    "watch_count": len(watch),
                }

        rem = max(0.0, deadline - time.time())
        if rem <= 0.0:
            break
        time.sleep(min(max(0.1, float(poll_interval_s)), rem))

    return {
        "triggered": False,
        "reason": "timeout",
        "watch_count": len(watch),
        "baseline_ready": len(baseline),
    }


def _wait_for_ws_trigger(*, args: argparse.Namespace, day_key: str, timeout_s: float) -> Dict[str, Any]:
    watch = _load_watch_tickers(
        day_key=day_key,
        watch_top_n=int(args.ws_watch_top_n),
        min_ev_dollars=float(args.orders_min_ev_dollars),
        max_close_hours=float(args.orders_max_close_hours),
    )
    poly_symbols, poly_symbol_map = _resolve_polymarket_symbols(
        market_tickers=watch,
        manual_csv=str(args.polymarket_symbols or ""),
        allow_manual_unaligned=bool(args.polymarket_allow_manual_unaligned),
    )
    has_poly = bool(args.polymarket_trigger) and bool(poly_symbols)

    if not watch and not has_poly:
        idle = min(max(0.0, float(timeout_s)), max(0.0, float(args.ws_no_tickers_sleep_seconds)))
        if idle > 0:
            time.sleep(idle)
        return {"triggered": False, "reason": "no_tickers", "watch_count": 0, "poly_watch_count": 0}

    try:
        out = asyncio.run(
            _await_any_market_trigger(
                kalshi_watch=watch,
                poly_symbols=poly_symbols,
                poly_symbol_to_kalshi_tickers=poly_symbol_map,
                timeout_s=float(timeout_s),
                kalshi_min_move_pct=float(args.ws_trigger_min_move_pct),
                kalshi_min_move_cents=float(args.ws_trigger_min_move_cents),
                kalshi_use_private_auth=bool(args.ws_use_private_auth),
                polymarket_enabled=bool(args.polymarket_trigger),
                polymarket_min_move_pct=float(args.polymarket_trigger_min_move_pct),
                polymarket_ws_url=str(args.polymarket_ws_url or ""),
            )
        )
        if "watch_count" not in out:
            out["watch_count"] = len(watch)
        out.setdefault("poly_watch_count", len(poly_symbols))
        kalshi_out = out.get("kalshi") if isinstance(out.get("kalshi"), dict) else {}
        kalshi_reason = str(kalshi_out.get("reason") or "")
        if (
            not bool(out.get("triggered"))
            and bool(args.ws_fallback_poll)
            and watch
            and kalshi_reason == "ws_error"
        ):
            poll_out = _poll_move_trigger(
                market_tickers=watch,
                timeout_s=float(timeout_s),
                min_move_pct=float(args.ws_trigger_min_move_pct),
                min_move_cents=float(args.ws_trigger_min_move_cents),
                poll_interval_s=float(args.ws_fallback_poll_interval_seconds),
            )
            if "watch_count" not in poll_out:
                poll_out["watch_count"] = len(watch)
            poll_out.setdefault("poly_watch_count", len(poly_symbols))
            poll_out.setdefault("fallback_from", str(kalshi_out.get("error") or "kalshi_ws_error"))
            poll_out.setdefault("prior_ws", out)
            return poll_out
        return out
    except Exception as exc:
        ws_err = {
            "triggered": False,
            "reason": "ws_error",
            "error": type(exc).__name__,
            "watch_count": len(watch),
            "poly_watch_count": len(poly_symbols),
        }
        if bool(args.ws_fallback_poll):
            poll_out = _poll_move_trigger(
                market_tickers=watch,
                timeout_s=float(timeout_s),
                min_move_pct=float(args.ws_trigger_min_move_pct),
                min_move_cents=float(args.ws_trigger_min_move_cents),
                poll_interval_s=float(args.ws_fallback_poll_interval_seconds),
            )
            if "watch_count" not in poll_out:
                poll_out["watch_count"] = len(watch)
            poll_out.setdefault("poly_watch_count", len(poly_symbols))
            poll_out.setdefault("fallback_from", ws_err.get("error"))
            return poll_out
        return ws_err


def _seconds_until_next_interval_boundary(interval_minutes: float) -> float:
    step = max(1.0, float(interval_minutes) * 60.0)
    now = time.time()
    next_boundary = (math.floor(now / step) + 1) * step
    return max(0.0, float(next_boundary - now))


def _write_status(payload: Dict[str, Any], *, tag: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", str(tag or "").strip()).strip("-")
    suffix = f"_{safe}" if safe else ""
    out_path = ROOT / "reports" / "ops" / f"live_status{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _acquire_lock(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    payload = {"pid": os.getpid(), "started_ts": _utc_now().isoformat()}
    os.write(fd, json.dumps(payload).encode("utf-8"))
    os.close(fd)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _clear_stale_lock(lock_path: Path) -> bool:
    if not lock_path.exists():
        return False
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    pid_raw = payload.get("pid")
    try:
        pid = int(pid_raw)
    except Exception:
        pid = 0

    # If pid is missing/invalid or process is gone, treat lock as stale.
    if pid <= 0 or not _pid_is_alive(pid):
        try:
            lock_path.unlink()
            return True
        except Exception:
            return False
    return False


def _run_cycle(args: argparse.Namespace, *, allow_order_placement: bool = True) -> Dict[str, Any]:
    day = _utc_day()
    cycle_ts = _utc_now().isoformat()
    tag = re.sub(r"[^A-Za-z0-9_-]+", "-", str(args.tag or "").strip()).strip("-")
    suffix = f"_{tag}" if tag else ""
    day_key = f"{day}{suffix}"

    commands: List[Dict[str, Any]] = []
    ok = True
    error_message = ""

    run_daily_cmd = [
        PY,
        str(ROOT / "orchestrator" / "run_daily.py"),
        "--config",
        str(args.config),
        "--date",
        day,
        "--tag",
        str(args.tag or ""),
    ]
    c_ok, rc, out, err = _run_cmd(run_daily_cmd)
    commands.append({"name": "run_daily", "cmd": run_daily_cmd, "ok": c_ok, "return_code": rc})
    if not c_ok:
        ok = False
        error_message = f"run_daily failed: rc={rc}"

    if ok:
        queue_cmd = [
            PY,
            str(ROOT / "orchestrator" / "build_research_queue.py"),
            "--date",
            day,
            "--tag",
            str(args.tag or ""),
            "--top-n",
            str(int(args.queue_top_n)),
            "--max-close-hours",
            str(float(args.queue_max_close_hours)),
            "--min-ev-dollars",
            str(float(args.queue_min_ev_dollars)),
        ]
        c_ok, rc, out, err = _run_cmd(queue_cmd)
        commands.append({"name": "build_research_queue", "cmd": queue_cmd, "ok": c_ok, "return_code": rc})
        if not c_ok:
            ok = False
            error_message = f"build_research_queue failed: rc={rc}"

    if ok and bool(args.hydrate_news):
        hyd_cmd = [
            PY,
            str(ROOT / "orchestrator" / "hydrate_news.py"),
            "--config",
            str(args.config),
            "--date",
            day,
            "--tag",
            str(args.tag or ""),
            "--as-of-ts",
            cycle_ts,
        ]
        if bool(args.hydrate_tavily):
            hyd_cmd.append("--tavily")
        c_ok, rc, out, err = _run_cmd(hyd_cmd)
        commands.append({"name": "hydrate_news", "cmd": hyd_cmd, "ok": c_ok, "return_code": rc})
        if not c_ok:
            ok = False
            error_message = f"hydrate_news failed: rc={rc}"

    if ok and allow_order_placement:
        place_cmd = [
            PY,
            str(ROOT / "orchestrator" / "place_live_orders.py"),
            "--config",
            str(args.config),
            "--date",
            day,
            "--tag",
            str(args.tag or ""),
            "--top-n",
            str(int(args.orders_top_n)),
            "--min-ev-dollars",
            str(float(args.orders_min_ev_dollars)),
            "--max-close-hours",
            str(float(args.orders_max_close_hours)),
            "--size-contracts",
            str(int(args.orders_size_contracts)),
        ]
        if bool(args.orders_fill_aware):
            place_cmd.append("--fill-aware")
        if bool(args.orders_cancel_stale):
            place_cmd.append("--cancel-stale")
        if bool(args.orders_cancel_replace):
            place_cmd.append("--cancel-replace")
        if bool(args.orders_unwind_unselected):
            place_cmd.append("--unwind-unselected")
            place_cmd.extend(["--unwind-max-contracts", str(int(args.orders_unwind_max_contracts))])
        if bool(args.orders_bankroll_aware):
            place_cmd.append("--bankroll-aware")
        if float(args.orders_bankroll_available_dollars) > 0:
            place_cmd.extend(["--bankroll-available-dollars", str(float(args.orders_bankroll_available_dollars))])
        place_cmd.extend(["--bankroll-balance-units", str(args.orders_bankroll_balance_units)])
        place_cmd.extend(["--bankroll-reserve-ratio", str(float(args.orders_bankroll_reserve_ratio))])
        place_cmd.extend(["--bankroll-max-total-pct", str(float(args.orders_bankroll_max_total_pct))])
        place_cmd.extend(["--bankroll-max-order-pct", str(float(args.orders_bankroll_max_order_pct))])
        place_cmd.extend(["--bankroll-min-order-dollars", str(float(args.orders_bankroll_min_order_dollars))])
        place_cmd.extend(["--bankroll-price-buffer-dollars", str(float(args.orders_bankroll_price_buffer_dollars))])
        place_cmd.extend(["--max-net-yes-contracts-per-ticker", str(int(args.orders_max_net_yes_contracts_per_ticker))])
        place_cmd.extend(["--max-buy-orders-per-ticker-per-day", str(int(args.orders_max_buy_orders_per_ticker_per_day))])
        place_cmd.extend(["--max-buy-orders-per-category-per-day", str(int(args.orders_max_buy_orders_per_category_per_day))])
        place_cmd.extend(["--max-new-notional-per-market", str(float(args.orders_max_new_notional_per_market))])
        place_cmd.extend(["--max-new-notional-per-category", str(float(args.orders_max_new_notional_per_category))])
        if bool(args.orders_no_derisk_negative_edge):
            place_cmd.append("--no-derisk-negative-edge")
        place_cmd.extend(["--derisk-edge-threshold-dollars", str(float(args.orders_derisk_edge_threshold_dollars))])
        place_cmd.extend(["--derisk-max-contracts-per-ticker", str(int(args.orders_derisk_max_contracts_per_ticker))])
        if bool(args.confirm):
            place_cmd.append("--confirm")
        c_ok, rc, out, err = _run_cmd(place_cmd)
        commands.append({"name": "place_live_orders", "cmd": place_cmd, "ok": c_ok, "return_code": rc})
        if not c_ok:
            ok = False
            error_message = f"place_live_orders failed: rc={rc}"

    status = {
        "timestamp": cycle_ts,
        "day": day,
        "day_key": day_key,
        "tag": tag,
        "last_run_status": "ok" if ok else "fail",
        "last_error": error_message,
        "orders": {
            "top_n": int(args.orders_top_n),
            "min_ev_dollars": float(args.orders_min_ev_dollars),
            "max_close_hours": float(args.orders_max_close_hours),
            "size_contracts": int(args.orders_size_contracts),
            "unwind_unselected": bool(args.orders_unwind_unselected),
            "unwind_max_contracts": int(args.orders_unwind_max_contracts),
            "bankroll_aware": bool(args.orders_bankroll_aware),
            "bankroll_available_dollars": float(args.orders_bankroll_available_dollars),
            "bankroll_balance_units": str(args.orders_bankroll_balance_units),
            "bankroll_reserve_ratio": float(args.orders_bankroll_reserve_ratio),
            "bankroll_max_total_pct": float(args.orders_bankroll_max_total_pct),
            "bankroll_max_order_pct": float(args.orders_bankroll_max_order_pct),
            "bankroll_min_order_dollars": float(args.orders_bankroll_min_order_dollars),
            "bankroll_price_buffer_dollars": float(args.orders_bankroll_price_buffer_dollars),
            "max_net_yes_contracts_per_ticker": int(args.orders_max_net_yes_contracts_per_ticker),
            "max_buy_orders_per_ticker_per_day": int(args.orders_max_buy_orders_per_ticker_per_day),
            "max_buy_orders_per_category_per_day": int(args.orders_max_buy_orders_per_category_per_day),
            "max_new_notional_per_market": float(args.orders_max_new_notional_per_market),
            "max_new_notional_per_category": float(args.orders_max_new_notional_per_category),
            "no_derisk_negative_edge": bool(args.orders_no_derisk_negative_edge),
            "derisk_edge_threshold_dollars": float(args.orders_derisk_edge_threshold_dollars),
            "derisk_max_contracts_per_ticker": int(args.orders_derisk_max_contracts_per_ticker),
            "allow_order_placement_this_cycle": bool(allow_order_placement),
            "confirm_flag": bool(args.confirm),
            "env_trading_enabled": str(os.environ.get("KALSHI_TRADING_ENABLED") or "").strip(),
        },
        "ws_trigger": {
            "enabled": bool(args.ws_trigger),
            "watch_top_n": int(args.ws_watch_top_n),
            "min_move_pct": float(args.ws_trigger_min_move_pct),
            "min_move_cents": float(args.ws_trigger_min_move_cents),
            "fallback_poll": bool(args.ws_fallback_poll),
            "fallback_poll_interval_seconds": float(args.ws_fallback_poll_interval_seconds),
        },
        "polymarket_trigger": {
            "enabled": bool(args.polymarket_trigger),
            "min_move_pct": float(args.polymarket_trigger_min_move_pct),
            "symbols": _parse_csv_values(str(args.polymarket_symbols or "")),
            "allow_manual_unaligned": bool(args.polymarket_allow_manual_unaligned),
            "ws_url": str(args.polymarket_ws_url or ""),
        },
        "commands": commands,
    }
    status_path = _write_status(status, tag=str(args.tag or ""))
    status["status_path"] = str(status_path)
    return status



def main() -> int:

    parser = argparse.ArgumentParser(description="Unattended LIVE daemon (daily ingest + queue + safety-gated live placement).")
    parser.add_argument("--config", default=str(ROOT / "config" / "defaults.json"), help="Path to config JSON (passed to run_daily/place_live_orders).")
    parser.add_argument("--interval-minutes", type=float, default=15.0, help="Loop interval in minutes (ignored with --once).")
    parser.add_argument("--align-to-interval", action="store_true", help="Sleep until the next wall-clock interval boundary before each cycle (e.g. with --interval-minutes 60, runs at :00).")
    parser.add_argument("--tag", default="", help="Optional output tag (passed through to run_daily/build_research_queue/place_live_orders/hydrate_news).")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit.")

    parser.add_argument("--queue-top-n", type=int, default=20, help="Queue size per cycle.")
    parser.add_argument("--queue-max-close-hours", type=float, default=72.0, help="Queue close horizon in hours.")
    parser.add_argument("--queue-min-ev-dollars", type=float, default=0.05, help="Queue min EV dollars.")

    parser.add_argument("--orders-top-n", type=int, default=5, help="Max orders per cycle.")
    parser.add_argument("--orders-min-ev-dollars", type=float, default=0.10, help="Min recomputed EV dollars to place.")
    parser.add_argument("--orders-max-close-hours", type=float, default=72.0, help="Order close horizon in hours.")
    parser.add_argument("--orders-size-contracts", type=int, default=10, help="Contracts per order.")
    parser.add_argument("--orders-fill-aware", action="store_true", help="Pass --fill-aware to place_live_orders.")
    parser.add_argument("--orders-cancel-stale", dest="orders_cancel_stale", action="store_true", help="Pass --cancel-stale to place_live_orders.")
    parser.add_argument("--no-orders-cancel-stale", dest="orders_cancel_stale", action="store_false", help="Disable --cancel-stale pass-through.")
    parser.add_argument("--orders-cancel-replace", dest="orders_cancel_replace", action="store_true", help="Pass --cancel-replace to place_live_orders.")
    parser.add_argument("--no-orders-cancel-replace", dest="orders_cancel_replace", action="store_false", help="Disable --cancel-replace pass-through.")
    parser.add_argument("--orders-unwind-unselected", dest="orders_unwind_unselected", action="store_true", help="Pass --unwind-unselected to place_live_orders.")
    parser.add_argument("--no-orders-unwind-unselected", dest="orders_unwind_unselected", action="store_false", help="Disable --unwind-unselected pass-through.")
    parser.add_argument("--orders-unwind-max-contracts", type=int, default=0, help="Pass --unwind-max-contracts to place_live_orders (0 means full detected position).")
    parser.add_argument("--orders-bankroll-aware", action="store_true", help="Pass --bankroll-aware to place_live_orders.")
    parser.add_argument("--orders-bankroll-available-dollars", type=float, default=0.0, help="Optional manual available-cash override for bankroll sizing.")
    parser.add_argument("--orders-bankroll-balance-units", choices=("auto", "dollars", "cents"), default="auto", help="Units hint for balance parsing in bankroll-aware mode.")
    parser.add_argument("--orders-bankroll-reserve-ratio", type=float, default=0.10, help="Reserve fraction of available cash to keep unallocated.")
    parser.add_argument("--orders-bankroll-max-total-pct", type=float, default=0.50, help="Max fraction of post-reserve cash deployable per cycle.")
    parser.add_argument("--orders-bankroll-max-order-pct", type=float, default=0.20, help="Max fraction of deployable cycle budget per order.")
    parser.add_argument("--orders-bankroll-min-order-dollars", type=float, default=1.0, help="Minimum estimated dollars per order after bankroll sizing.")
    parser.add_argument("--orders-bankroll-price-buffer-dollars", type=float, default=0.01, help="Safety per-contract cost buffer for bankroll sizing.")
    parser.add_argument("--orders-max-net-yes-contracts-per-ticker", type=int, default=10, help="Pass --max-net-yes-contracts-per-ticker to place_live_orders.")
    parser.add_argument("--orders-max-buy-orders-per-ticker-per-day", type=int, default=1, help="Pass --max-buy-orders-per-ticker-per-day to place_live_orders.")
    parser.add_argument("--orders-max-buy-orders-per-category-per-day", type=int, default=5, help="Pass --max-buy-orders-per-category-per-day to place_live_orders.")
    parser.add_argument("--orders-max-new-notional-per-market", type=float, default=0.0, help="Pass --max-new-notional-per-market to place_live_orders (<=0 uses config cap).")
    parser.add_argument("--orders-max-new-notional-per-category", type=float, default=0.0, help="Pass --max-new-notional-per-category to place_live_orders (<=0 uses config cap).")
    parser.add_argument("--orders-no-derisk-negative-edge", action="store_true", help="Pass --no-derisk-negative-edge to place_live_orders.")
    parser.add_argument("--orders-derisk-edge-threshold-dollars", type=float, default=0.0, help="Pass --derisk-edge-threshold-dollars to place_live_orders.")
    parser.add_argument("--orders-derisk-max-contracts-per-ticker", type=int, default=0, help="Pass --derisk-max-contracts-per-ticker to place_live_orders (0 means full eligible size).")

    parser.add_argument("--confirm", action="store_true", help="Pass through to place_live_orders.py (still requires KALSHI_TRADING_ENABLED=1).")
    parser.add_argument("--hydrate-news", action="store_true", help="Run hydrate_news.py each cycle (default: off).")
    parser.add_argument("--hydrate-tavily", action="store_true", help="Enable Tavily search in hydrate_news.py (requires TAVILY_API_KEY).")
    parser.add_argument("--ws-trigger", action="store_true", help="Between scheduled cycles, monitor Kalshi WS ticker updates and trigger an immediate re-run on large price moves.")
    parser.add_argument("--ws-watch-top-n", type=int, default=20, help="Max watched tickers per WS session (selected tickets first, then top EV candidates).")
    parser.add_argument("--ws-trigger-min-move-pct", type=float, default=10.0, help="Trigger re-run when |yes_price move| / baseline >= this percent.")
    parser.add_argument("--ws-trigger-min-move-cents", type=float, default=0.0, help="Optional absolute cents move trigger (0 disables absolute trigger).")
    parser.add_argument("--ws-use-private-auth", action="store_true", help="Use authenticated WS connection (requires private key env vars).")
    parser.add_argument("--no-ws-fallback-poll", action="store_true", help="Disable fallback REST polling when WS is unavailable.")
    parser.add_argument("--ws-fallback-poll-interval-seconds", type=float, default=15.0, help="Polling interval for WS fallback mode.")
    parser.add_argument("--ws-no-tickers-sleep-seconds", type=float, default=30.0, help="Sleep backoff when no watch tickers are available.")
    parser.add_argument("--ws-error-backoff-seconds", type=float, default=15.0, help="Sleep backoff after WS errors before retrying.")
    parser.add_argument("--trigger-place-orders", action="store_true", help="Allow WS/Polymarket triggered cycles to place orders (default: analyze-only on triggers).")
    parser.add_argument("--polymarket-trigger", action="store_true", help="Enable Polymarket RTDS crypto price trigger as an additional re-check source.")
    parser.add_argument("--polymarket-trigger-min-move-pct", type=float, default=2.0, help="Trigger re-run when Polymarket RTDS crypto price move exceeds this percent.")
    parser.add_argument("--polymarket-symbols", default="", help="Optional comma-separated Polymarket RTDS symbols (e.g., btcusdt,ethusdt). Empty=auto-infer from watched Kalshi tickers.")
    parser.add_argument("--polymarket-allow-manual-unaligned", action="store_true", help="Allow manual Polymarket symbols not aligned to watched Kalshi tickers (default: false).")
    parser.add_argument("--polymarket-ws-url", default="", help="Optional Polymarket RTDS websocket URL override.")
    parser.set_defaults(
        orders_cancel_stale=True,
        orders_cancel_replace=True,
        orders_unwind_unselected=True,
    )
    args = parser.parse_args()
    args.ws_fallback_poll = not bool(args.no_ws_fallback_poll)

    args.config = str(Path(str(args.config)).resolve())
    if float(args.interval_minutes) <= 0:
        print("--interval-minutes must be > 0")
        return 2
    if int(args.orders_unwind_max_contracts) < 0:
        print("--orders-unwind-max-contracts must be >= 0")
        return 2
    if float(args.orders_bankroll_available_dollars) < 0:
        print("--orders-bankroll-available-dollars must be >= 0")
        return 2
    if not (0.0 <= float(args.orders_bankroll_reserve_ratio) <= 0.95):
        print("--orders-bankroll-reserve-ratio must be in [0, 0.95]")
        return 2
    if not (0.0 <= float(args.orders_bankroll_max_total_pct) <= 1.0):
        print("--orders-bankroll-max-total-pct must be in [0, 1]")
        return 2
    if not (0.0 <= float(args.orders_bankroll_max_order_pct) <= 1.0):
        print("--orders-bankroll-max-order-pct must be in [0, 1]")
        return 2
    if float(args.orders_bankroll_min_order_dollars) < 0:
        print("--orders-bankroll-min-order-dollars must be >= 0")
        return 2
    if float(args.orders_bankroll_price_buffer_dollars) < 0:
        print("--orders-bankroll-price-buffer-dollars must be >= 0")
        return 2
    if int(args.orders_max_net_yes_contracts_per_ticker) < 0:
        print("--orders-max-net-yes-contracts-per-ticker must be >= 0")
        return 2
    if int(args.orders_max_buy_orders_per_ticker_per_day) < 0:
        print("--orders-max-buy-orders-per-ticker-per-day must be >= 0")
        return 2
    if int(args.orders_max_buy_orders_per_category_per_day) < 0:
        print("--orders-max-buy-orders-per-category-per-day must be >= 0")
        return 2
    if float(args.orders_max_new_notional_per_market) < 0:
        print("--orders-max-new-notional-per-market must be >= 0")
        return 2
    if float(args.orders_max_new_notional_per_category) < 0:
        print("--orders-max-new-notional-per-category must be >= 0")
        return 2
    if int(args.orders_derisk_max_contracts_per_ticker) < 0:
        print("--orders-derisk-max-contracts-per-ticker must be >= 0")
        return 2
    if int(args.ws_watch_top_n) <= 0:
        print("--ws-watch-top-n must be > 0")
        return 2
    if float(args.ws_trigger_min_move_pct) < 0:
        print("--ws-trigger-min-move-pct must be >= 0")
        return 2
    if float(args.ws_trigger_min_move_cents) < 0:
        print("--ws-trigger-min-move-cents must be >= 0")
        return 2
    if float(args.ws_no_tickers_sleep_seconds) < 0 or float(args.ws_error_backoff_seconds) < 0:
        print("WS backoff values must be >= 0")
        return 2
    if float(args.ws_fallback_poll_interval_seconds) <= 0:
        print("--ws-fallback-poll-interval-seconds must be > 0")
        return 2
    if float(args.polymarket_trigger_min_move_pct) < 0:
        print("--polymarket-trigger-min-move-pct must be >= 0")
        return 2

    safe_tag = re.sub(r"[^A-Za-z0-9_-]+", "-", str(args.tag or "").strip()).strip("-")
    suffix = f"_{safe_tag}" if safe_tag else ""
    lock_path = ROOT / "data" / "ops" / f"live_daemon{suffix}.lock"
    try:
        _acquire_lock(lock_path)
    except FileExistsError:
        if _clear_stale_lock(lock_path):
            try:
                _acquire_lock(lock_path)
            except FileExistsError:
                print(f"live daemon lock exists: {lock_path}")
                return 1
        else:
            print(f"live daemon lock exists: {lock_path}")
            return 1

    try:
        if bool(args.align_to_interval):
            delay = _seconds_until_next_interval_boundary(float(args.interval_minutes))
            if delay > 0:
                time.sleep(delay)

        while True:
            status = _run_cycle(args, allow_order_placement=True)
            print(json.dumps(status, indent=2))
            if args.once:
                break

            if bool(args.ws_trigger) or bool(args.polymarket_trigger):
                if bool(args.align_to_interval):
                    deadline = time.time() + _seconds_until_next_interval_boundary(float(args.interval_minutes))
                else:
                    deadline = time.time() + max(1.0, float(args.interval_minutes) * 60.0)

                while time.time() < deadline:
                    remaining = max(0.0, deadline - time.time())
                    if remaining <= 0.0:
                        break
                    day_key = str(status.get("day_key") or _utc_day())
                    ws_result = _wait_for_ws_trigger(args=args, day_key=day_key, timeout_s=remaining)
                    if bool(ws_result.get("triggered")):
                        print(json.dumps({"ws_trigger": ws_result}, indent=2))
                        status = _run_cycle(args, allow_order_placement=bool(args.trigger_place_orders))
                        print(json.dumps(status, indent=2))
                        continue
                    if str(ws_result.get("reason") or "") == "ws_error":
                        backoff = min(max(0.0, float(args.ws_error_backoff_seconds)), max(0.0, deadline - time.time()))
                        if backoff > 0:
                            time.sleep(backoff)
                        continue
                    break
            else:
                if bool(args.align_to_interval):
                    delay = _seconds_until_next_interval_boundary(float(args.interval_minutes))
                    if delay > 0:
                        time.sleep(delay)
                else:
                    sleep_seconds = max(1.0, float(args.interval_minutes) * 60.0)
                    time.sleep(sleep_seconds)
        return 0
    finally:
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

