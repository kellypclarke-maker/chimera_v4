from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence
from urllib.parse import urlparse

import websockets
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from .kalshi_public import DEFAULT_MAIN_HOST, canonical_kalshi_base


def _default_ws_url() -> str:
    u = (os.environ.get("KALSHI_WS_URL") or os.environ.get("KALSHI_WS_BASE") or "").strip()
    if u:
        return u.rstrip("/")

    rest_in = (
        os.environ.get("KALSHI_API_BASE")
        or os.environ.get("KALSHI_PUBLIC_BASE")
        or os.environ.get("KALSHI_BASE")
        or ""
    )
    rest = canonical_kalshi_base(str(rest_in).strip())
    parsed = urlparse(rest)
    host = parsed.netloc or DEFAULT_MAIN_HOST
    return f"wss://{host}/trade-api/ws/v2"


def _load_private_key_pem(*, private_key_pem: Optional[bytes], private_key_path: Optional[Path]) -> bytes:
    if private_key_pem:
        return private_key_pem
    raw = (os.environ.get("KALSHI_API_PRIVATE_KEY") or "").strip()
    if raw:
        return raw.encode("utf-8")
    p = (str(private_key_path) if private_key_path is not None else (os.environ.get("KALSHI_PRIVATE_KEY_PATH") or "")).strip()
    if p:
        return Path(p).read_bytes()
    raise ValueError("missing Kalshi private key (KALSHI_API_PRIVATE_KEY or KALSHI_PRIVATE_KEY_PATH)")


def _ws_auth_headers(*, key_id: str, private_key_pem: bytes, path: str = "/trade-api/ws/v2") -> Dict[str, str]:
    kid = str(key_id or "").strip()
    if not kid:
        raise ValueError("missing KALSHI_API_KEY_ID")

    ts = str(int(time.time() * 1000.0))
    message = ts + "GET" + str(path)
    key = load_pem_private_key(private_key_pem, password=None)
    sig = key.sign(
        message.encode("utf-8"),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": kid,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode("ascii"),
    }


@dataclass(frozen=True)
class WsSubscribeResult:
    request_id: int
    channels: List[str]
    market_tickers: List[str]


class KalshiWsClient:
    def __init__(
        self,
        *,
        ws_url: Optional[str] = None,
        use_private_auth: bool = False,
        key_id: Optional[str] = None,
        private_key_pem: Optional[bytes] = None,
        private_key_path: Optional[Path] = None,
    ) -> None:
        self.ws_url = (ws_url or _default_ws_url()).rstrip("/")
        self.use_private_auth = bool(use_private_auth)
        self._key_id = (key_id or os.environ.get("KALSHI_API_KEY_ID") or "").strip()
        self._private_key_pem = None if not self.use_private_auth else _load_private_key_pem(private_key_pem=private_key_pem, private_key_path=private_key_path)
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._next_id = 1

    async def connect(self) -> None:
        if self._ws is not None:
            return
        headers = None
        if self.use_private_auth:
            assert self._private_key_pem is not None
            parsed = urlparse(self.ws_url)
            path = parsed.path or "/trade-api/ws/v2"
            headers = _ws_auth_headers(key_id=self._key_id, private_key_pem=self._private_key_pem, path=path)
            print(
                f"[KALSHI_WS] connecting url={self.ws_url} private_auth=1 "
                f"key_id_present={bool(self._key_id)} header_keys={sorted(headers.keys())}"
            )
        else:
            print(f"[KALSHI_WS] connecting url={self.ws_url} private_auth=0")
        self._ws = await websockets.connect(
            self.ws_url,
            additional_headers=headers,
            ping_interval=None,
            close_timeout=5,
            open_timeout=10,
        )

    async def close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            finally:
                self._ws = None

    async def send(self, payload: Dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("WS not connected")
        await self._ws.send(json.dumps(payload, separators=(",", ":"), sort_keys=True))

    async def subscribe(
        self,
        *,
        channels: Sequence[str],
        market_ticker: Optional[str] = None,
        market_tickers: Sequence[str] = (),
        send_initial_snapshot: bool = True,
    ) -> WsSubscribeResult:
        ch = [str(c).strip() for c in channels if str(c).strip()]
        if not ch:
            raise ValueError("channels required")

        mt = [str(t).strip().upper() for t in market_tickers if str(t).strip()]
        single = str(market_ticker).strip().upper() if market_ticker else ""
        params: Dict[str, Any] = {"channels": ch}
        if single:
            params["market_ticker"] = single
        elif mt:
            params["market_tickers"] = mt
        if send_initial_snapshot:
            params["send_initial_snapshot"] = True

        req_id = self._next_id
        self._next_id += 1
        await self.send({"id": req_id, "cmd": "subscribe", "params": params})
        return WsSubscribeResult(request_id=req_id, channels=ch, market_tickers=([single] if single else mt))

    async def recv_json(self) -> Dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("WS not connected")
        raw = await self._ws.recv()
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="ignore")
        msg = json.loads(str(raw))
        if not isinstance(msg, dict):
            raise TypeError("unexpected WS message (expected JSON object)")
        return msg

    async def iter_messages(self) -> AsyncIterator[Dict[str, Any]]:
        while True:
            yield await self.recv_json()


async def ws_collect_ticker_snapshot(
    *,
    market_tickers: Sequence[str],
    ws_url: Optional[str] = None,
    use_private_auth: bool = False,
    timeout_s: float = 15.0,
) -> Dict[str, Dict[str, Any]]:
    raw_want = {str(t).strip().upper() for t in market_tickers if str(t).strip()}
    want: set[str] = set()
    dropped_parent_events: List[str] = []
    for t in sorted(raw_want):
        is_parent_sports_event = (t.startswith("KXNBAGAME-") or t.startswith("KXNHLGAME-")) and t.count("-") == 1
        if is_parent_sports_event:
            dropped_parent_events.append(t)
            continue
        want.add(t)
    if dropped_parent_events:
        print(
            f"[KALSHI_WS] dropping parent sports event tickers from subscribe set "
            f"count={len(dropped_parent_events)} tickers={dropped_parent_events[:5]}"
        )
    if not want:
        return {}
    client = KalshiWsClient(ws_url=ws_url, use_private_auth=use_private_auth)
    print(
        f"[KALSHI_WS] snapshot start url={client.ws_url} tickers={len(want)} "
        f"timeout_s={float(timeout_s):.1f}"
    )
    await client.connect()
    try:
        await client.subscribe(channels=["ticker"], market_tickers=sorted(want), send_initial_snapshot=True)
        out: Dict[str, Dict[str, Any]] = {}
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline and len(out) < len(want):
            try:
                msg = await asyncio.wait_for(client.recv_json(), timeout=max(0.1, deadline - time.time()))
            except asyncio.TimeoutError:
                continue
            if str(msg.get("type") or "").strip().lower() != "ticker":
                continue
            m = msg.get("msg") if isinstance(msg.get("msg"), dict) else {}
            mt = str(m.get("market_ticker") or "").strip().upper()
            if mt in want:
                out[mt] = m
        return out
    finally:
        await client.close()
