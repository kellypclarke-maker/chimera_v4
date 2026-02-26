import requests
import os
from chimera.clients.kalshi import KalshiPrivateClient
from chimera.trading.crypto_oracle_v4 import CryptoOracle
from chimera.trading.weather_oracle_v4 import WeatherOracle
import asyncio

async def test():
    client = KalshiPrivateClient(
        key_id=os.environ.get('KALSHI_API_KEY_ID'),
        private_key_path=os.environ.get('KALSHI_PRIVATE_KEY_PATH')
    )

    btc_t = "BVOL-26FEB25-T70000"
    wx_t = "KXWEATHER-26FEB25NYC"

    # Get kalshi markets
    try:
        btc_book = client.request("GET", f"/markets/{btc_t}")["market"]
        print(f"[{btc_t}] Kalshi YES Ask: {btc_book.get('yes_ask')}c | YES Bid: {btc_book.get('yes_bid')}c")
    except Exception:
        print(f"[{btc_t}] Could not fetch Kalshi book.")

    try:
        wx_book = client.request("GET", f"/markets/{wx_t}")["market"]
        print(f"[{wx_t}] Kalshi YES Ask: {wx_book.get('yes_ask')}c | YES Bid: {wx_book.get('yes_bid')}c")
    except Exception:
        print(f"[{wx_t}] Could not fetch Kalshi book.")

    print("\n--- SHADOW ORACLE TELEMETRY ---")
    crypto_oracle = CryptoOracle("btcusdt", kalshi_strike_price=90000.0, flash_threshold=50.0)
    weather_oracle = WeatherOracle("NYC")
    
    # Run briefly to get one point of data
    task = asyncio.create_task(crypto_oracle.listen_binance_ws())
    await asyncio.sleep(2)
    task.cancel()
    
    if crypto_oracle.spot_price:
        p_true_btc = crypto_oracle.calculate_p_true(24.0, 0.5)
        print(f"[Crypto Oracle] BTC Spot: ${crypto_oracle.spot_price:,.2f} -> Shadow P_True: {p_true_btc:.4f}")
    else:
        print("[Crypto Oracle] Waiting on WS data...")

    print(f"[Weather Oracle] Base Volatility (Sigma): {weather_oracle.base_sigma}")
    print("[Weather Oracle] (Live polling requires extended run for NOAA cache drops)")

if __name__ == "__main__":
    asyncio.run(test())
