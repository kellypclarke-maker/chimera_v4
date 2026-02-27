import os
from chimera.clients.kalshi import KalshiPrivateClient
import requests

client = KalshiPrivateClient(
    key_id=os.environ.get('KALSHI_API_KEY_ID'),
    private_key_path=os.environ.get('KALSHI_PRIVATE_KEY_PATH')
)

ticker = "KXNBAGAME-26FEB25OKCDET-OKC"

# 1. Get current best bid
try:
    market_resp = client.request("GET", f"/markets/{ticker}")
    market = market_resp.get("market", {})
    best_bid = market.get('yes_bid', 0)
    print(f"📉 Best Bid for {ticker}: {best_bid}")
except Exception as e:
    print(f"Error getting market: {e}")
    best_bid = 20

# 2. Price at 1 cent below best bid
price_cents = max(1, best_bid - 1)
count = 1

# 3. Simulate autotrade payload
clean_count = int(count)
clean_price = int(round(price_cents))
print(f"[DEBUG] Sending Order Payload: ticker={ticker}, side=yes, action=buy, count={clean_count}, price_cents={clean_price}")

# 4. Place order and catch verbose error
try:
    resp = client.place_order(
        ticker=ticker,
        side="yes",
        action="buy",
        count=clean_count,
        price_cents=clean_price
    )
    print(f"✅ Order Accepted: {resp}")
except requests.exceptions.HTTPError as e:
    print(f"❌ HTTP Error placing order: {e}")
    print(f"   Response Body: {e.response.text}")
except Exception as e:
    print(f"❌ Unknown Error: {e}")
