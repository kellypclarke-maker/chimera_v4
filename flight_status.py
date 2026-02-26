import os
from chimera.clients.kalshi import KalshiPrivateClient

client = KalshiPrivateClient(
    key_id=os.environ.get('KALSHI_API_KEY_ID'),
    private_key_path=os.environ.get('KALSHI_PRIVATE_KEY_PATH')
)

try:
    balance_resp = client.request("GET", "/portfolio/balance")
    balance = balance_resp.get("balance", 0)
    print(f"💰 Account Balance: ${balance / 100:.2f}")
except Exception as e:
    try:
        balance_resp = client._request("GET", "/portfolio/balance")
        balance = balance_resp.get("balance", 0)
        print(f"💰 Account Balance: ${balance / 100:.2f}")
    except Exception as e2:
        print(f"Error getting balance: {e2}")

try:
    pos_resp = client.request("GET", "/portfolio/positions")
    market_positions = pos_resp.get("market_positions", [])
    active_positions = [mp for mp in market_positions if mp.get("position", 0) > 0]
    
    if not active_positions:
        print("📊 Active Positions: 0 (Steel Gate holding)")
    else:
        print("📊 Active Positions:")
        for mp in active_positions:
            print(f"  - {mp['ticker']}: {mp['position']} contracts")
except Exception as e:
    try:
        pos_resp = client._request("GET", "/portfolio/positions")
        market_positions = pos_resp.get("market_positions", [])
        active_positions = [mp for mp in market_positions if mp.get("position", 0) > 0]
        
        if not active_positions:
            print("📊 Active Positions: 0 (Steel Gate holding)")
        else:
            print("📊 Active Positions:")
            for mp in active_positions:
                print(f"  - {mp['ticker']}: {mp['position']} contracts")
    except Exception as e2:
        print(f"Error getting positions: {e2}")
