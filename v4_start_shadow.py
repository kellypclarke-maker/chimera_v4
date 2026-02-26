import asyncio
import time
from chimera.trading.autotrade import ExecutionEngine

class ShadowExecutionEngine(ExecutionEngine):
    def __init__(self, tickers, edge_threshold_cents):
        super().__init__(tickers, edge_threshold_cents)
        
        class MockClient:
            def cancel_order(self, order_id):
                print(f"[SHADOW] Cancelled order {order_id}")
            def place_order(self, ticker, side, action, count, price_cents):
                order_id = f"shadow-{int(time.time())}"
                print(f"[SHADOW] Placed order {order_id} for {ticker} at {price_cents}c")
                return {"order_id": order_id}
                
        self.client = MockClient()
        # Simulate our current position of 42 contracts discovered from the REST API
        self.filled_contracts = {"KXNBAGAME-26FEB25OKCDET-OKC": 42}

    async def test_order(self):
        ticker = "KXNBAGAME-26FEB25OKCDET-OKC"
        
        # Manually set a profitable state
        self.dynamic_p_true[ticker] = 0.60
        self.kalshi_order_book[ticker] = {"yes_bid": 50, "yes_ask": 55}
        
        print("Evaluating order with max_position = 1 and current_position = 42")
        await self._evaluate_order(ticker)

async def main():
    engine = ShadowExecutionEngine(tickers=["KXNBAGAME-26FEB25OKCDET-OKC"], edge_threshold_cents=5.0)
    await engine.test_order()

if __name__ == "__main__":
    asyncio.run(main())
