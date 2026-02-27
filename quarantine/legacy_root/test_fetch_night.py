import requests
base = "https://api.elections.kalshi.com/trade-api/v2"

# Find a valid BVOL ticker
r = requests.get(f"{base}/events?limit=200&status=open&series_ticker=KXBVOL")
events = r.json().get("events", [])
for e in events:
    r2 = requests.get(f"{base}/markets?event_ticker={e.get('event_ticker')}&limit=1")
    for m in r2.json().get("markets", []):
        print("Valid Crypto Market:", m.get("ticker"))

# Find a valid Weather ticker
r3 = requests.get(f"{base}/events?limit=200&status=open&series_ticker=KXHIGHM")
events = r3.json().get("events", [])
for e in events:
    r4 = requests.get(f"{base}/markets?event_ticker={e.get('event_ticker')}&limit=1")
    for m in r4.json().get("markets", []):
        if "NYC" in m.get("ticker", ""):
            print("Valid Weather Market:", m.get("ticker"))
