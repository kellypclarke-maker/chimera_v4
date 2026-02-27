#!/usr/bin/env python3
import asyncio
import argparse
from chimera.trading.autotrade import ExecutionEngine
from chimera.trading.crypto_oracle_v4 import CryptoOracle
from chimera.trading.nba_live_v4 import NBALiveOracle
import time
import os

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-contracts", type=int, default=1)
    parser.add_argument("--force-size-contracts", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=1.5)
    args = parser.parse_args()

    tickers = ["KXNBAGAME-26FEB25OKCDET-OKC", "BVOL-26FEB25-T70000", "KXNHLGAME-26FEB25VGKLA-VGK"]
    
    # We subclass ExecutionEngine to route the oracles correctly
    class LiveEngine(ExecutionEngine):
        def __init__(self, tickers, edge_threshold_cents):
            super().__init__(tickers, edge_threshold_cents)
            # Enforce micro size
            for mo in self.managed_orders.values():
                mo.remaining_count = args.size_contracts

        async def oracle_listener(self):
            print("Starting Triple Oracles (NBA, Crypto, NHL)...")
            from chimera.trading.nhl_live_v4 import NHLLiveOracle
            
            nba_oracle = NBALiveOracle(
                espn_game_id="401584688",
                pre_game_p_true_home=0.60,
                base_edge=self.edge_threshold_cents
            )
            crypto_oracle = CryptoOracle("btcusdt", kalshi_strike_price=90000.0, flash_threshold=50.0)
            nhl_oracle = NHLLiveOracle(
                espn_game_id="401560370", # Placeholder ESPN game ID
                pre_game_p_true_home=0.55,
                base_edge=self.edge_threshold_cents
            )
            
            asyncio.create_task(nba_oracle.poll_espn())
            asyncio.create_task(crypto_oracle.listen_binance_ws())
            asyncio.create_task(nhl_oracle.poll_espn())

            ping_counter = 0

            while True:
                await asyncio.sleep(args.poll_seconds)
                ping_counter += args.poll_seconds
                
                async with self.lock:
                    # NBA Eval
                    nba_t = "KXNBAGAME-26FEB25OKCDET-OKC"
                    dynamic_p_home, edge_modifier = nba_oracle.calculate_live_p_true()
                    # Simulating a massive probability crash to test the Exit Valve
                    dynamic_p_home = 0.05 
                    self.dynamic_p_true[nba_t] = dynamic_p_home
                    await self._evaluate_order(nba_t)
                    
                    # Crypto Eval
                    btc_t = "BVOL-26FEB25-T70000"
                    dynamic_btc = 0.5 # Default baseline if websocket hasn't initialized
                    if crypto_oracle.spot_price is not None:
                        dynamic_btc = crypto_oracle.calculate_p_true(time_to_expiry_hours=24.0, volatility=0.5)
                        self.dynamic_p_true[btc_t] = dynamic_btc
                        if crypto_oracle.mass_cancel_flag:
                            await self._mass_cancel(btc_t)
                        else:
                            await self._evaluate_order(btc_t)
                            
                    # NHL Eval
                    nhl_t = "KXNHLGAME-26FEB25VGKLA-VGK"
                    dynamic_nhl_p, nhl_edge_modifier = nhl_oracle.calculate_live_p_true()
                    self.dynamic_p_true[nhl_t] = dynamic_nhl_p
                    await self._evaluate_order(nhl_t)
                    
                    # 60s Ping Test
                    if ping_counter >= 60.0:
                        nhl_book = self.kalshi_order_book.get(nhl_t, {})
                        print(f"[PING] NHL ({nhl_t}): p_true={dynamic_nhl_p:.4f} | yes_ask={nhl_book.get('yes_ask')}c")
                        btc_book = self.kalshi_order_book.get(btc_t, {})
                        print(f"[PING] Crypto ({btc_t}): p_true={dynamic_btc:.4f} | yes_ask={btc_book.get('yes_ask')}c")
                        ping_counter = 0

    engine = LiveEngine(tickers=tickers, edge_threshold_cents=5.0)
    await engine.run()

if __name__ == "__main__":
    asyncio.run(main())
