from chimera.clients.kalshi import KalshiPrivateClient
import time
import json

client = KalshiPrivateClient()
ticker = "KXNHLGAME-26FEB25VGKLA-VGK"

# 1. Get current position
try:
    pos_resp = client.request("GET", "/portfolio/positions", params={"ticker": ticker})
    market_positions = pos_resp.get("market_positions", [])
    position = 0
    for mp in market_positions:
        if mp["ticker"] == ticker:
            position = mp["position"]
            break
except Exception as e:
    print(f"Error getting position: {e}")
    position = 42 # Fallback to known value

print(f"Current position for {ticker}: {position}")

if position > 0:
    # 2. Get current orderbook to find best bid
    try:
        ob_resp = client.request("GET", f"/markets/{ticker}/orderbook")
        bids = ob_resp.get("orderbook", {}).get("yes", [])
        best_bid = bids[0][0] if bids else 1 # Default to 1 cent if empty
        
        # Price aggressively to sell immediately
        sell_price = max(1, best_bid - 5) 
        print(f"Best bid is {best_bid}c. Placing aggressive limit sell at {sell_price}c")
        
        # 3. Place sell order
        print(f"Placing sell order for {position} contracts at {sell_price}c...")
        order_resp = client.place_order(
            ticker=ticker,
            side="yes",
            action="sell",
            count=position,
            price_cents=sell_price
        )
        print(f"Order placed: {order_resp}")
        
        # Wait a moment for the order to fill
        time.sleep(3)
        
        # 4. Verify position again
        pos_resp = client.request("GET", "/portfolio/positions", params={"ticker": ticker})
        market_positions = pos_resp.get("market_positions", [])
        new_position = 0
        for mp in market_positions:
            if mp["ticker"] == ticker:
                new_position = mp["position"]
                break
        print(f"New position for {ticker}: {new_position}")
        
    except Exception as e:
        print(f"Error during liquidation: {e}")
else:
    print("No position to liquidate.")
