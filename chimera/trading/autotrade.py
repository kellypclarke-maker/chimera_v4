from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional

from chimera.clients.kalshi import KalshiPrivateClient
from chimera.clients.kalshi_ws import KalshiWsClient
from chimera.fees import expected_value_yes

@dataclass
class ManagedOrder:
    ticker: str
    order_id: Optional[str] = None
    active: bool = True
    limit_cents: Optional[int] = None
    remaining_count: int = 1


class ExecutionEngine:
    def __init__(self, tickers: List[str], edge_threshold_cents: float):
        self.tickers = tickers
        self.edge_threshold_cents = edge_threshold_cents
        self.dynamic_p_true: Dict[str, float] = {}
        self.kalshi_order_book: Dict[str, Dict[str, Optional[int]]] = {
            t: {"yes_bid": None, "yes_ask": None} for t in tickers
        }
        self.managed_orders: Dict[str, ManagedOrder] = {t: ManagedOrder(ticker=t) for t in tickers}
        self.filled_contracts: Dict[str, int] = {t: 0 for t in tickers}
        self.client = KalshiPrivateClient()
        self.lock = asyncio.Lock()

    def _calculate_max_limit_cents(self, p_true: float) -> int:
        """
        Mandate #4: Anti-Spoofing Limit Logic
        Limit prices must be anchored strictly to our dynamic p_true minus a required edge margin.
        We do NOT rely on yes_bid + 1.
        """
        max_price_cents = (p_true * 100.0) - self.edge_threshold_cents
        return max(1, min(99, int(max_price_cents)))

    async def _cancel_order(self, ticker: str):
        mo = self.managed_orders.get(ticker)
        if mo is None or not mo.active or not mo.order_id:
            return
        oid = str(mo.order_id)
        try:
            self.client.cancel_order(order_id=oid)
            mo.order_id = None
            mo.limit_cents = None
            print(f"[KILL SWITCH] Canceled order for {ticker}")
        except Exception as e:
            print(f"[ERROR] Cancel failed for {oid}: {e}")

    async def _mass_cancel(self, ticker: Optional[str] = None):
        """
        Immediately mass-cancel resting orders if the market moves against us.
        """
        for t in self.managed_orders:
            if ticker is None or ticker == t:
                await self._cancel_order(t)

    async def _evaluate_order(self, ticker: str):
        """
        Evaluate whether to place or reprice an order based on current p_true and Kalshi order book.
        """
        p_true = self.dynamic_p_true.get(ticker)
        if p_true is None:
            return

        book = self.kalshi_order_book[ticker]
        mo = self.managed_orders[ticker]
        filled_contracts = int(self.filled_contracts.get(ticker, 0))
        
        def get_cat(t: str) -> str:
            tu = t.upper()
            if "NBA" in tu: return "NBA"
            if "NHL" in tu: return "NHL"
            if "BVOL" in tu or "BTC" in tu or "ETH" in tu: return "CRYPTO"
            return "WEATHER"
            
        cat_filled = sum(c for t, c in self.filled_contracts.items() if get_cat(t) == get_cat(ticker))
        max_position = 1

        # Exit Valve:
        # If we already hold a YES position, evaluate whether we should emergency-exit at bid.
        if filled_contracts > 0:
            bid_raw = book.get("yes_bid")
            try:
                bid_cents = int(bid_raw) if bid_raw is not None else None
            except (TypeError, ValueError):
                bid_cents = None

            if bid_cents is not None and 1 <= bid_cents <= 99:
                bid_price = bid_cents / 100.0
                hold_ev = expected_value_yes(
                    p_true=p_true,
                    price=bid_price,
                    fee=0.0,
                    adverse_selection_discount=0.05,
                )
                p_true_unlikely = float(p_true) <= 0.45
                if hold_ev < -0.10 or p_true_unlikely:
                    print(
                        f"[EXIT VALVE] Triggered for {ticker}: hold_ev={hold_ev:.4f}, "
                        f"p_true={float(p_true):.4f}, bid={bid_cents}c, filled={filled_contracts}"
                    )
                    try:
                        # Clear resting buy order before forcing exit.
                        await self._cancel_order(ticker)
                        self.client.place_order(
                            ticker=ticker,
                            side="yes",
                            action="sell",
                            count=int(filled_contracts),
                            price_cents=int(round(bid_cents)),
                        )
                        self.filled_contracts[ticker] = 0
                        print(f"[EXIT VALVE] Submitted sell for {ticker} at {bid_cents}c; position reset")
                    except Exception as e:
                        print(f"[ERROR] Exit Valve sell failed for {ticker}: {e}")
                    return

        ask = book.get("yes_ask")
        if ask is None or ask <= 1:
            return

        # Anti-Spoofing Repricing: strictly bound our limit price by our max allowable calculation
        max_limit_cents = self._calculate_max_limit_cents(p_true)
        
        # We do not cross the spread (ask - 1) and we never bid higher than our max_limit_cents
        limit_cents = min(max_limit_cents, ask - 1)

        if limit_cents < 1:
            await self._mass_cancel(ticker)
            return

        price = limit_cents / 100.0
        # Incorporate adverse selection discount via fees.py new signature
        ev = expected_value_yes(p_true=p_true, price=price, fee=0.0, adverse_selection_discount=0.05)
        
        # If EV drops below our minimum threshold, execute mass-cancel
        if ev <= 0.0:
            await self._mass_cancel(ticker)
            
            # The Exit Valve: Liquidate active positions if edge is dead
            filled = self.filled_contracts.get(ticker, 0)
            if filled > 0:
                print(f"[EXIT] Negative EV detected for {ticker}. Liquidating {filled} contracts.")
                try:
                    self.client.place_order(
                        ticker=ticker,
                        side="yes",
                        action="sell",
                        count=filled,
                        price_cents=1
                    )
                    self.filled_contracts[ticker] = 0
                except Exception as e:
                    print(f"[ERROR] Exit Valve failed: {e}")
            return

        # Hold Rule: cancel only when filled position is at/above cap.
        if cat_filled >= max_position:
            await self._cancel_order(ticker)
            return

        # Quiet Rule: when we have no fills and already have a resting order, wait for fill.
        if cat_filled == 0 and mo.order_id:
            return
            
        # Place or replace order at the new strict limit
        try:
            if mo.order_id:
                try:
                    self.client.cancel_order(order_id=mo.order_id)
                except Exception as e:
                    print(f"[WARN] Cancel failed for {mo.order_id}: {e}")
                finally:
                    mo.order_id = None
            
            clean_count = int(mo.remaining_count)
            clean_price = int(round(limit_cents))
            
            resp = self.client.place_order(
                ticker=ticker,
                side="yes",
                action="buy",
                count=clean_count,
                price_cents=clean_price
            )
            mo.order_id = resp.get("order", {}).get("order_id") or resp.get("order_id")
            mo.limit_cents = clean_price
            print(f"[OK] Placed order for {ticker} at {clean_price}c")
        except Exception as e:
            print(f"[ERROR] Failed to place order for {ticker}: {e}")

    async def oracle_listener(self):
        """
        Stream A (The Oracle): External data source.
        Listens to external source, updates dynamic_p_true, triggers kill switches.
        """
        print("Starting Oracle Listener (Stream A)...")
        # Placeholder for external WS connection (e.g., Binance, ESPN)
        while True:
            # 1. Await external message
            await asyncio.sleep(0.1) 
            
            # Simulated update
            ticker = self.tickers[0] if self.tickers else None
            if not ticker:
                continue
                
            new_p_true = 0.55 # Placeholder for specific NBA/Crypto logic
            
            async with self.lock:
                # 2. Immediately recalculate dynamic p_true
                self.dynamic_p_true[ticker] = new_p_true
                
                # 3. Dynamic Kill Switch (Mandate #1 & #3)
                max_limit = self._calculate_max_limit_cents(new_p_true)
                if max_limit < 1:
                    await self._mass_cancel(ticker)
                else:
                    price = max_limit / 100.0
                    ev = expected_value_yes(p_true=new_p_true, price=price, fee=0.0, adverse_selection_discount=0.05)
                    if ev <= 0.0:
                        await self._mass_cancel(ticker)
                    else:
                        await self._evaluate_order(ticker)

    async def exchange_listener(self):
        """
        Stream B (The Exchange): Kalshi WebSocket.
        Listens to order book updates, applies Anti-Spoofing Repricing.
        """
        print("Starting Exchange Listener (Stream B)...")
        while True:
            ws = KalshiWsClient(use_private_auth=True)
            try:
                await ws.connect()
                # Also subscribe to 'fill' events so we can track our positions correctly
                await ws.subscribe(channels=["ticker", "fill"], market_tickers=self.tickers)

                timeout_counter = 0
                heartbeat_counter = 0
                while True:
                    # 1. Await Kalshi updates
                    try:
                        msg = await asyncio.wait_for(ws.recv_json(), timeout=30.0)
                        timeout_counter = 0 # Reset on success
                        
                        msg_type = msg.get("type")
                        if msg_type != "ticker":
                            if heartbeat_counter % 20 == 0:
                                print(f"[Exchange] Non-ticker message received: {msg_type}")
                            heartbeat_counter += 1
                    except asyncio.TimeoutError:
                        timeout_counter += 1
                        if timeout_counter >= 3:
                            print("[Exchange Listener] 3 consecutive timeouts; breaking to reconnect...")
                            break
                        else:
                            print(f"[Exchange Listener] WS read timeout ({timeout_counter}/3); continuing to wait...")
                            continue

                    if msg.get("type") == "ticker":
                        payload = msg.get("msg", {})
                        ticker = payload.get("market_ticker")
                        if not ticker or ticker not in self.kalshi_order_book:
                            continue

                        async with self.lock:
                            # 2. Evaluate new order book
                            book = self.kalshi_order_book[ticker]
                            book["yes_bid"] = payload.get("yes_bid")
                            book["yes_ask"] = payload.get("yes_ask")

                            # 3. Anti-Spoofing Repricing (Mandate #4)
                            await self._evaluate_order(ticker)
                    
                    elif msg.get("type") == "fill":
                        payload = msg.get("msg", {})
                        ticker = payload.get("ticker")
                        if ticker in self.filled_contracts:
                            async with self.lock:
                                # Example structure - assuming "count" provides filled contracts
                                count = payload.get("count", 1) 
                                self.filled_contracts[ticker] += count
                                print(f"[Exchange] Fill recorded for {ticker}! Total position: {self.filled_contracts[ticker]}")
                                # Evaluate to ensure we respect position limits going forward
                                await self._evaluate_order(ticker)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[Exchange Listener] WebSocket error: {e}")
            finally:
                try:
                    await ws.close()
                except Exception as e:
                    print(f"[Exchange Listener] WS close error: {e}")
            await asyncio.sleep(1.0)

    async def run(self):
        await asyncio.gather(
            self.oracle_listener(),
            self.exchange_listener()
        )

async def run_autotrade(*args, **kwargs):
    # This is the entry point that the rest of the application will call.
    # The specific arguments would be adapted to the new Engine inputs.
    engine = ExecutionEngine(tickers=[], edge_threshold_cents=5.0)
    await engine.run()
