from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from urllib.parse import urlencode, urlparse

import requests
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from .kalshi_public import canonical_kalshi_base


def _public_base() -> str:
    return canonical_kalshi_base()


class KalshiPrivateReadClient:
    """Signed private client with read-only helpers. No order placement methods in this phase."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        key_id: Optional[str] = None,
        private_key_pem: Optional[bytes] = None,
        private_key_path: Optional[Path] = None,
        timestamp_unit: str = "ms",
        padding_mode: str = "pss",
        sign_query: bool = False,
    ) -> None:
        self.base_url = canonical_kalshi_base(base_url or os.environ.get("KALSHI_API_BASE") or _public_base()).rstrip("/")
        self.key_id = (key_id or os.environ.get("KALSHI_API_KEY_ID") or "").strip()
        self.timestamp_unit = str(timestamp_unit).strip().lower() or "ms"
        self.padding_mode = str(padding_mode).strip().lower() or "pss"
        self.sign_query = bool(sign_query)

        parsed = urlparse(self.base_url)
        self._base_path = (parsed.path or "").rstrip("/") or "/trade-api/v2"
        if not self.key_id:
            raise ValueError("missing KALSHI_API_KEY_ID")

        pem: Optional[bytes] = private_key_pem
        if pem is None:
            raw = os.environ.get("KALSHI_API_PRIVATE_KEY") or ""
            if raw.strip():
                pem = raw.encode("utf-8")
        if pem is None and private_key_path is not None:
            pem = Path(private_key_path).read_bytes()
        if pem is None:
            p = os.environ.get("KALSHI_PRIVATE_KEY_PATH") or ""
            if p.strip():
                pem = Path(p.strip()).read_bytes()
        if pem is None:
            raise ValueError("missing Kalshi private key (KALSHI_API_PRIVATE_KEY or KALSHI_PRIVATE_KEY_PATH)")

        self._key = load_pem_private_key(pem, password=None)
        self._session = requests.Session()

    def _timestamp(self) -> str:
        now = time.time()
        return str(int(now)) if self.timestamp_unit == "s" else str(int(now * 1000.0))

    def _pad(self):
        if self.padding_mode == "pss":
            return padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH)
        return padding.PKCS1v15()

    def _sign(self, message: str) -> str:
        sig = self._key.sign(message.encode("utf-8"), self._pad(), hashes.SHA256())
        return base64.b64encode(sig).decode("ascii")

    def _headers(self, method: str, path: str) -> Dict[str, str]:
        ts = self._timestamp()
        prehash = ts + method.upper() + path
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": self._sign(prehash),
            "Content-Type": "application/json",
        }

    def request(self, method: str, path: str, *, params: Optional[Dict[str, object]] = None, json_body: Any = None) -> Dict[str, Any]:
        p = "/" + str(path).lstrip("/")
        qs = ""
        if params:
            items = sorted((str(k), str(v)) for k, v in params.items())
            qs = urlencode(items)

        sign_path = self._base_path + p + (f"?{qs}" if (qs and self.sign_query) else "")
        url = self.base_url + p + (f"?{qs}" if qs else "")
        body = "" if json_body is None else json.dumps(json_body, separators=(",", ":"), sort_keys=True)
        headers = self._headers(method, sign_path)

        resp = self._session.request(method.upper(), url, data=(body if body else None), headers=headers, timeout=30.0)
        resp.raise_for_status()
        out = resp.json()
        if not isinstance(out, dict):
            raise TypeError(f"unexpected response from {p} (expected object)")
        return out

    def _request_first_success(
        self,
        method: str,
        paths: Sequence[str],
        *,
        params: Optional[Dict[str, object]] = None,
        json_body: Any = None,
    ) -> Dict[str, Any]:
        last_exc: Optional[BaseException] = None
        for p in paths:
            try:
                return self.request(method, p, params=params, json_body=json_body)
            except requests.HTTPError as exc:
                last_exc = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status == 404:
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("no paths provided")

    def get_portfolio_balance(self) -> Dict[str, Any]:
        return self._request_first_success("GET", ["/portfolio/balance", "/portfolio"], params=None)

    def get_portfolio_positions(self) -> Dict[str, Any]:
        return self._request_first_success("GET", ["/portfolio/positions"], params=None)

    def get_fills(self, *, ticker: Optional[str] = None, limit: int = 500, cursor: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, object] = {"limit": int(limit)}
        if ticker:
            params["ticker"] = str(ticker).strip().upper()
        if cursor:
            params["cursor"] = str(cursor)
        return self._request_first_success("GET", ["/portfolio/fills", "/fills"], params=params)

class KalshiPrivateClient(KalshiPrivateReadClient):
    """
    Signed private client with order placement helpers.

    Safety note: call sites must gate live trading (e.g. CLI --confirm + env KALSHI_TRADING_ENABLED=1).
    """

    def get_orders(self, *, status: Optional[str] = None, limit: int = 200, cursor: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, object] = {"limit": int(limit)}
        if status:
            params["status"] = str(status).strip().lower()
        if cursor:
            params["cursor"] = str(cursor)
        return self._request_first_success("GET", ["/portfolio/orders", "/orders"], params=params)

    def place_order(
        self,
        *,
        ticker: str,
        side: str,
        action: str,
        count: int,
        price_cents: int,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        side_norm = str(side).strip().lower()
        if side_norm not in {"yes", "no"}:
            raise ValueError("side must be 'yes' or 'no'")
        act = str(action).strip().lower()
        if act not in {"buy", "sell"}:
            raise ValueError("action must be 'buy' or 'sell'")
        px = int(price_cents)
        if px < 1 or px > 99:
            raise ValueError("price_cents must be in [1, 99]")
        qty = int(count)
        if qty <= 0:
            raise ValueError("count must be > 0")

        payload: Dict[str, Any] = {
            "ticker": str(ticker).strip().upper(),
            "side": side_norm,
            "action": act,
            "count": qty,
            "type": "limit",
        }
        if client_order_id:
            payload["client_order_id"] = str(client_order_id).strip()

        if side_norm == "yes":
            payload["yes_price"] = px
        else:
            payload["no_price"] = px

        return self._request_first_success("POST", ["/portfolio/orders", "/orders"], json_body=payload)

    def cancel_order(self, *, order_id: str) -> Dict[str, Any]:
        oid = str(order_id).strip()
        if not oid:
            raise ValueError("order_id required")
        return self._request_first_success("DELETE", [f"/portfolio/orders/{oid}", f"/orders/{oid}"], params=None)

