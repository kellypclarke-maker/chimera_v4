import asyncio
import os
import requests
import time
from chimera.clients.kalshi import fetch_market
from chimera.trading.crypto_oracle_v4 import CryptoOracle
from chimera.trading.weather_oracle_v4 import WeatherOracle

async def get_crypto_p_true():
    # Use the same parameters as in run_shadow_fleet.py
    oracle = CryptoOracle("btcusdt", kalshi_strike_price=90000.0, flash_threshold=50.0)
    task = asyncio.create_task(oracle.listen_binance_ws())
    
    print("Waiting for Binance spot price...")
    start_time = time.time()
    while oracle.spot_price is None and time.time() - start_time < 5:
        await asyncio.sleep(0.5)
    
    if oracle.spot_price is None:
        print("WS failed, trying REST fallback...")
        import requests
        try:
            r = requests.get("https://api.binance.us/api/v3/ticker/price?symbol=BTCUSDT", timeout=5)
            if r.status_code == 200:
                oracle.spot_price = float(r.json()["price"])
        except Exception as e:
            print(f"REST fallback failed: {e}")

    if oracle.spot_price is not None:
        # Use a realistic strike if 90000 is too far
        # For status, let's use a strike closer to spot for better resolution in the report
        oracle.kalshi_strike_price = 70000.0
        p_true = oracle.calculate_p_true(time_to_expiry_hours=24.0, volatility=0.5)
        task.cancel()
        return oracle.spot_price, p_true, oracle.kalshi_strike_price
    task.cancel()
    return None, None, None

async def get_weather_sigma():
    oracle = WeatherOracle("NYC")
    import aiohttp
    async with aiohttp.ClientSession() as session:
        headers = {"User-Agent": "Chimera-v4-Status/0.1"}
        url = "https://api.weather.gov/gridpoints/OKX/33,35/forecast/hourly"
        async with session.get(url, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                oracle._extract_temperatures(data)
                # Use current time for the window
                now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
                next_hour_iso = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).isoformat()
                sigma = oracle.calculate_dynamic_sigma(now_iso, next_hour_iso)
                return sigma
    return None

import datetime as dt

def get_kalshi_price(ticker):
    session = requests.Session()
    try:
        m = fetch_market(ticker=ticker, session=session)
        if m:
            yes_bid = m.get("yes_bid")
            yes_ask = m.get("yes_ask")
            return yes_bid, yes_ask
    except Exception as e:
        print(f"Error fetching Kalshi price for {ticker}: {e}")
    return None, None

async def main():
    print("--- Night Watch Status Update (Corrected Tickers) ---")
    
    # Corrected Tickers
    date_token = "26FEB26"
    wx_base = f"KXHIGHNY-{date_token}"
    btc_base = f"KXBTC-{date_token}"
    
    # Weather Brackets: 40 to 41, 42 to 43
    # Based on discovered tickers: KXHIGHNY-26FEB26-B40.5 is "Between 40 and 41"
    wx_40_41 = f"{wx_base}-B40.5"
    wx_42_43 = f"{wx_base}-B42.5"
    
    # For BTC, let us skip Kalshi price if we cannot find the ticker
    for ticker in [wx_40_41, wx_42_43]:
        bid, ask = get_kalshi_price(ticker)
        print(f"Kalshi {ticker}: Bid {bid}c, Ask {ask}c")
    
    # Oracles
    spot, crypto_p, strike = await get_crypto_p_true()
    wx_sigma = await get_weather_sigma()
    
    if spot:
        print(f"Crypto Oracle: BTC Spot ${spot:.2f} (Strike ${strike}) -> p_true (KXBTC): {crypto_p:.4f}")
    else:
        print("Crypto Oracle: Failed to fetch spot price.")
        
    if wx_sigma:
        print(f"Weather Oracle: NYC Dynamic Sigma: {wx_sigma:.2f}")
    else:
        print("Weather Oracle: Failed to fetch forecast.")

if __name__ == "__main__":
    asyncio.run(main())
