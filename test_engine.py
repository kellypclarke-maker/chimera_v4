import asyncio
from chimera.trading.autotrade import ExecutionEngine
from chimera.trading.nba_live_v4 import NBALiveOracle
from chimera.fees import FeeConfig, expected_value_yes

async def run():
    engine = ExecutionEngine(tickers=["KXNBAGAME-26FEB25OKCDET", "KXNBAGAME-26FEB25SASTOR"], edge_threshold_cents=5.0)
    
    # Mock Kalshi Order Book
    engine.kalshi_order_book["KXNBAGAME-26FEB25OKCDET"] = {"yes_bid": 50, "yes_ask": 55}
    engine.kalshi_order_book["KXNBAGAME-26FEB25SASTOR"] = {"yes_bid": 50, "yes_ask": 55}

    # Mock NBALiveOracle
    engine.oracle = NBALiveOracle(
        espn_game_id="401584688",
        pre_game_p_true_home=0.60,
        base_edge=engine.edge_threshold_cents
    )
    
    # Calculate EV directly to log
    dynamic_p_home, edge_modifier = engine.oracle.calculate_live_p_true()
    
    price = 50 / 100.0
    fee_config = FeeConfig.from_env()
    ev = expected_value_yes(
        p_true=dynamic_p_home, 
        price=price, 
        adverse_selection_discount=0.05,
        is_maker=False,
        fee_config=fee_config,
    )
    print(f"[{engine.oracle.__class__.__name__}] Calculating EV for NBA Market:")
    print(f"Dynamic P_true: {dynamic_p_home:.4f}")
    print(f"Price: {price:.2f}")
    print(f"FeeConfig: {fee_config}")
    print(f"Calculated EV: {ev:.4f}")

asyncio.run(run())
