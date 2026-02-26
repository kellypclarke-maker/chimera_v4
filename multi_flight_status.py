import os
from chimera.clients.kalshi import KalshiPrivateClient
import time
import math

def get_cat(t: str) -> str:
    tu = t.upper()
    if "NBA" in tu: return "NBA"
    if "NHL" in tu: return "NHL"
    if "BVOL" in tu or "BTC" in tu or "ETH" in tu: return "CRYPTO"
    return "WEATHER"

client = KalshiPrivateClient(
    key_id=os.environ.get('KALSHI_API_KEY_ID'),
    private_key_path=os.environ.get('KALSHI_PRIVATE_KEY_PATH')
)

def run_report():
    try:
        pos_resp = client.request("GET", "/portfolio/positions")
    except Exception as e:
        try:
            pos_resp = client._request("GET", "/portfolio/positions")
        except Exception as e2:
            print(f"Error getting positions: {e2}")
            return
            
    market_positions = pos_resp.get("market_positions", [])
    active_positions = [mp for mp in market_positions if mp.get("position", 0) > 0]
    
    cats = {"NBA": None, "NHL": None, "CRYPTO": None, "WEATHER": None}
    
    for mp in active_positions:
        cat = get_cat(mp["ticker"])
        if cats.get(cat) is None:
            cats[cat] = {
                "ticker": mp["ticker"],
                "position": mp["position"],
                "pnl": mp.get("realized_pnl_dollars", "0.00")
            }
            
    print("🚀 FLIGHT STATUS REPORT 🚀")
    print("")
    for cat in ["NBA", "NHL", "CRYPTO", "WEATHER"]:
        data = cats[cat]
        if data:
            print(f"{cat}: {data['ticker']} | {data['position']} Contracts | PNL: ${data['pnl']}")
        else:
            print(f"{cat}: NONE | 0 Contracts | PNL: $0.00")

run_report()
