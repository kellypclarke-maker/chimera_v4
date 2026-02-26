from chimera.clients.kalshi import KalshiPrivateClient
import json

client = KalshiPrivateClient()

# 1. Fetch resting orders and cancel them
try:
    orders_resp = client.request("GET", "/portfolio/orders", params={"status": "resting"})
    orders = orders_resp.get("orders", [])
    print(f"Found {len(orders)} resting orders.")
    for o in orders:
        client.cancel_order(order_id=o["order_id"])
        print(f"Canceled order {o['order_id']}")
except Exception as e:
    # If the method is _request, try that
    try:
        orders_resp = client._request("GET", "/portfolio/orders", params={"status": "resting"})
        orders = orders_resp.get("orders", [])
        print(f"Found {len(orders)} resting orders.")
        for o in orders:
            client.cancel_order(order_id=o["order_id"])
            print(f"Canceled order {o['order_id']}")
    except Exception as e2:
        print("Could not fetch/cancel orders:", e2)

# 2. Fetch positions for the ticker
ticker = "KXNBAGAME-26FEB25OKCDET-OKC"
try:
    pos_resp = client.request("GET", f"/portfolio/positions", params={"ticker": ticker})
    # or just fetching all positions
    print("Positions Response:", json.dumps(pos_resp))
except Exception as e:
    try:
        pos_resp = client._request("GET", f"/portfolio/positions/{ticker}")
        print("Position:", json.dumps(pos_resp))
    except Exception as e2:
        print("Could not fetch position:", e2)
