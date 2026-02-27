import os
from chimera.clients.kalshi import KalshiPrivateClient

client = KalshiPrivateClient(
    key_id=os.environ.get('KALSHI_API_KEY_ID'),
    private_key_path=os.environ.get('KALSHI_PRIVATE_KEY_PATH')
)

# 1. Check Balance
try:
    balance_resp = client.request("GET", "/portfolio/balance")
    balance = balance_resp.get("balance", 0)
    print(f"💰 Current Balance: ${balance / 100:.2f}")
except Exception as e:
    # Try alternate method if get_balance is not directly available or different response format
    try:
        balance_resp = client._request("GET", "/portfolio/balance")
        balance = balance_resp.get("balance", 0)
        print(f"💰 Current Balance: ${balance / 100:.2f}")
    except Exception as e2:
        print(f"Error getting balance: {e2}")

# 2. Check Ticker Status
ticker = "KXNBAGAME-26FEB25OKCDET-OKC"
try:
    market_resp = client.request("GET", f"/markets/{ticker}")
    market = market_resp.get("market", {})
    print(f"📊 Market Status for {ticker}: {market.get('status')}")
    print(f"📉 Best Bid: {market.get('yes_bid')}, Best Ask: {market.get('yes_ask')}")
except Exception as e:
    try:
        market_resp = client._request("GET", f"/markets/{ticker}")
        market = market_resp.get("market", {})
        print(f"📊 Market Status for {ticker}: {market.get('status')}")
        print(f"📉 Best Bid: {market.get('yes_bid')}, Best Ask: {market.get('yes_ask')}")
    except Exception as e2:
        print(f"Error getting market: {e2}")
