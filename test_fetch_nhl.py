import requests
base = "https://api.elections.kalshi.com/trade-api/v2"
r = requests.get(f"{base}/events?limit=200&status=open&series_ticker=KXNHLGAME")
events = r.json().get("events", [])
for e in events:
    print("Found Event:", e.get("event_ticker"))
    r2 = requests.get(f"{base}/markets?event_ticker={e.get('event_ticker')}&limit=5")
    for m in r2.json().get("markets", []):
        print("  Market:", m.get("ticker"))
