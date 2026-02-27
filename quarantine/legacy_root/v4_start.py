import asyncio
import time
from chimera.trading.autotrade import ExecutionEngine
from chimera.trading.crypto_oracle_v4 import CryptoOracle

class ShadowExecutionEngine(ExecutionEngine):
    def __init__(self, tickers, edge_threshold_cents):
        super().__init__(tickers, edge_threshold_cents)
        
        # Override client methods for shadow mode to avoid real API calls
        class MockClient:
            def cancel_order(self, order_id):
                print(f"[SHADOW] Cancelled order {order_id}")
            def place_order(self, ticker, side, action, count, price_cents):
                order_id = f"shadow-{int(time.time())}"
                print(f"[SHADOW] Placed order {order_id} for {ticker} at {price_cents}c")
                return {"order_id": order_id}
                
        self.client = MockClient()

    async def oracle_listener(self):
        print("Starting Crypto Oracle Listener (Stream A)...")
        ticker = self.tickers[0] if self.tickers else "BVOL-26FEB25-T70000"
        
        # 1. Initialize the Crypto Oracle for BTC
        # Assuming current BTC price is around $90,000
        # flash_threshold is modified to 1.0 in the source code
        self.oracle = CryptoOracle("btcusdt", kalshi_strike_price=90000.0, flash_threshold=0.01)
        
        # Start the WebSocket listener in the background
        asyncio.create_task(self.oracle.listen_binance_ws())

        while True:
            # Sync the core loop to a fast evaluation rate
            await asyncio.sleep(0.5) 
            
            if self.oracle.spot_price is None:
                continue
                
            async with self.lock:
                # 2. Extract the Live Probability
                # Pass arbitrary time/volatility for test
                dynamic_p_true = self.oracle.calculate_p_true(time_to_expiry_hours=24.0, volatility=0.5)
                self.dynamic_p_true[ticker] = dynamic_p_true
                
                # 3. Dynamic Kill Switch Check
                if self.oracle.mass_cancel_flag:
                    print(f"[ENGINE] Mass cancel flag detected! Halting and pulling all orders for {ticker}.")
                    await self._mass_cancel(ticker)
                    # We can exit after detecting the kill switch for this test
                    return
                else:
                    # Proceed with normal evaluation
                    max_limit = self._calculate_max_limit_cents(dynamic_p_true)
                    if max_limit < 1:
                        await self._mass_cancel(ticker)
                    else:
                        await self._evaluate_order(ticker)

async def main():
    # Provide the BTC volatility ticker
    engine = ShadowExecutionEngine(tickers=["BVOL-26FEB25-T70000"], edge_threshold_cents=5.0)
    
    # Mock some initial Kalshi Order Book data so it can evaluate and "place" a shadow order
    engine.kalshi_order_book["BVOL-26FEB25-T70000"] = {"yes_bid": 40, "yes_ask": 60}
    
    # Since we want to exit early, we can use an asyncio Event or just let it run
    try:
        await asyncio.wait_for(engine.run(), timeout=15.0)
    except asyncio.TimeoutError:
        print("[TEST] Timeout reached.")

if __name__ == "__main__":
    asyncio.run(main())
