# Debugging Kalshi API/WS Changes and Fixes

The root issue is that Kalshi’s V2 API now *distinguishes “event” tickers from “market” tickers*, and our bot was calling the wrong endpoint. In the old flow, the bot treated every ticker like an **event** (calling GET `/trade-api/v2/events/{ticker}`) even when the ticker was actually a specific market. This causes 404 errors whenever the ticker is a market ID. For example, the log shows repeated 404s for `KXNHLGAME-26FEB26NYRWSH` – that’s a **market** ticker (a specific Rangers vs Washington outcome), not an event ID. Kalshi’s V2 API requires that *event* tickers be fetched via `/events/{event_ticker}`, whereas *market* tickers must be fetched via `/markets/{market_ticker}`. In short, a market ticker like `…NYRWSH` will never be found under the `/events` endpoint, hence the 404s.  

The fix is to bypass the event lookup whenever we know the ticker is already a market. In practice, when the user supplies an exact market ticker override (the ones with both teams, e.g. `KXNBAGAME-26FEB26MIAPHI-MIA`), we should **skip** the `_expand_parent_sports_event_to_market_tickers` discovery and *directly* call the GET `/markets/{market_ticker}` endpoint for that ticker. This “hard bypass” means no 404 from treating it as an event. In the bot code, that implies detecting override tickers (those with two teams or ending in the outcome code) and setting them as market IDs. We then call `get_markets` (or equivalent) with `endpoint="/markets/{ticker}"` directly, rather than trying to expand via the parent event. 

Likewise for the WebSocket: the subscription channel changed. In the old code, we likely subscribed using an `event_ticker` or generic channel, but Kalshi V2’s WS *expects a “market_ticker” channel for live market data*. The fix is to send a subscribe message that includes `"channel": "market_ticker"` and `"market_ticker": "<THE_EXACT_TICKER>"`. For example, the JSON should be:  
```
{ "id": 1, "cmd": "subscribe", "params": {
    "channel": "market_ticker",
    "market_ticker": "KXNHLGAME-26FEB26NYRWSH"
}}
```  
This ensures the socket will stream updates for that exact market. The symptom in logs (“WS snapshot received=0”) indicates that the old subscribe message was not matching any channel, so no data came. Using the correct channel and field fixes the WS feed. 

Finally, we must ensure category metadata is preserved. When we bypass discovery for an override ticker, the bot still needs to know it’s a sports event (NBA/NHL). Previously, discovery would tag events by category (sports vs crypto vs weather). If we skip discovery, we must manually assign the category. For example, if the override ticker matches the “KXNBA” prefix, set `category="NBA"` (and similarly for NHL). This way, the rest of the engine still treats it as a sports market. Without this, the override markets were getting no category and being ignored (`open_sports=0`). 

In summary: update the code so that **explicit market tickers use the `/markets` API and WS channel directly**, skipping the generic event path. Clear any cached “ghost” event entries for those tickers. And when bypassing discovery, manually tag the market with its category. These changes align the bot with the new Kalshi V2 API/WS requirements, restoring successful market lookups and data feeds.