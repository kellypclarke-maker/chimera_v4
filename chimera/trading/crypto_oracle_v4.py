import asyncio
import json
import time
import math
from collections import deque
from typing import Optional, Tuple, Deque
import websockets


def generate_viable_btc_tickers(
    spot_price: float,
    target_date: str,
    variance_pct: float = 0.02,
) -> list[str]:
    """
    Build a focused list of KXBTC daily tickers around current spot.

    Example output ticker:
      KXBTC-26FEB26-T68500
    """
    spot = float(spot_price)
    if spot <= 0.0:
        raise ValueError("spot_price must be positive")

    var = float(variance_pct)
    if var < 0.0:
        raise ValueError("variance_pct must be non-negative")

    date_token = str(target_date).strip().upper()
    if not date_token:
        raise ValueError("target_date is required")

    increment = 500
    lower_bound = spot * (1.0 - var)
    upper_bound = spot * (1.0 + var)

    first_strike = int(math.ceil(lower_bound / increment) * increment)
    last_strike = int(math.floor(upper_bound / increment) * increment)

    if first_strike > last_strike:
        nearest = int(round(spot / increment) * increment)
        return [f"KXBTC-{date_token}-T{nearest}"]

    return [
        f"KXBTC-{date_token}-T{strike}"
        for strike in range(first_strike, last_strike + increment, increment)
    ]


class CryptoOracle:
    """
    High-Frequency Crypto Oracle that connects directly to the Binance WebSocket
    aggregated trade stream to track real-time spot prices. Implements a Velocity 
    Kill Switch to instantly flag mass-cancel events during flash crashes.
    """
    def __init__(self, symbol: str, kalshi_strike_price: float, flash_threshold: float = 1.0):
        # Symbol should be lowercase for Binance WS, e.g., 'btcusdt'
        self.symbol = symbol.lower()
        self.kalshi_strike_price = kalshi_strike_price
        self.flash_threshold = flash_threshold
        
        self.ws_url = f"wss://stream.binance.us:9443/ws/{self.symbol}@aggTrade"
        
        # State
        self.spot_price: Optional[float] = None
        self.mass_cancel_flag: bool = False
        
        # Velocity Tracker: stores tuples of (timestamp, price)
        # We only need to keep the last ~5 seconds of data to check the 2-second window
        self.tick_history: Deque[Tuple[float, float]] = deque(maxlen=20000)
        self.history_window_seconds = 5.0

    async def listen_binance_ws(self):
        """
        The Spot WebSocket: Connects to Binance aggregated trade stream.
        No REST polling allowed; relies on real-time push events.
        """
        while True:
            try:
                async with websockets.connect(self.ws_url) as websocket:
                    print(f"[Crypto Oracle] Connected to Binance WS: {self.symbol}")
                    async for message in websocket:
                        data = json.loads(message)
                        # 'p' is the price in the aggTrade payload
                        price_str = data.get('p')
                        if price_str:
                            current_price = float(price_str)
                            current_time = time.monotonic()
                            
                            self.spot_price = current_price
                            self._update_velocity_tracker(current_time, current_price)
                            
            except Exception as e:
                print(f"[Crypto Oracle] WebSocket disconnected or error: {e}")
                # Brief backoff before reconnecting
                await asyncio.sleep(1.0)

    def _update_velocity_tracker(self, current_time: float, current_price: float):
        """
        The Velocity Kill Switch (Momentum Tracker):
        Maintains a rolling window of tick data and checks for flash crashes.
        """
        # 1. Add new tick
        self.tick_history.append((current_time, current_price))
        
        # 2. Prune data older than the total history window (5 seconds)
        while self.tick_history and (current_time - self.tick_history[0][0]) > self.history_window_seconds:
            self.tick_history.popleft()
            
        # 3. Check for massive movement within a 2-second window
        # We look back through the retained history for prices that occurred within the last 2 seconds.
        # Since history is pruned to 5 seconds, we iterate backwards.
        max_price_in_window = current_price
        min_price_in_window = current_price
        
        for tick_time, tick_price in reversed(self.tick_history):
            if (current_time - tick_time) <= 2.0:
                if tick_price > max_price_in_window:
                    max_price_in_window = tick_price
                if tick_price < min_price_in_window:
                    min_price_in_window = tick_price
            else:
                # We've moved past the 2-second window boundary
                break
                
        price_swing = max_price_in_window - min_price_in_window
        
        if price_swing >= self.flash_threshold:
            self.mass_cancel_flag = True
            # Optional: Log the trigger for analytics
            # print(f"[KILL SWITCH] Flash threshold exceeded: {price_swing:.2f} swing in 2s")
        else:
            self.mass_cancel_flag = False

    @property
    def distance_to_strike(self) -> Optional[float]:
        """
        Calculates the real-time distance from the spot price to the Kalshi strike.
        """
        if self.spot_price is None:
            return None
        return self.spot_price - self.kalshi_strike_price

    def calculate_p_true(self, time_to_expiry_hours: float, volatility: float) -> float:
        """
        Dynamic Probability Mapper:
        A simplified placeholder returning a baseline probability based on distance to the strike.
        Heavily penalizes if the mass_cancel_flag is active (flash crash detected).
        """
        if self.spot_price is None:
            return 0.5 # Default unknown state
            
        dist = self.distance_to_strike
        if dist is None:
            return 0.5
        
        # Basic placeholder logic:
        # If spot is exactly at strike, p_true = 0.5
        # We use a logistic function to scale the distance into a probability
        # The wider the volatility and longer the time, the flatter the curve.
        
        # Denominator prevents division by zero and scales based on volatility/time
        scale_factor = max(0.1, volatility * math.sqrt(time_to_expiry_hours))
        
        # Calculate baseline probability (logistic curve)
        # Assuming the market is "Will it resolve ABOVE the strike?"
        z_score = max(-500.0, min(500.0, dist / scale_factor))
        baseline_p_true = 1.0 / (1.0 + math.exp(-z_score))
        
        # Penalize if velocity kill switch is active
        if self.mass_cancel_flag:
            # If the market is crashing/spiking violently, we zero out the probability 
            # or push it to an extreme to guarantee a mass-cancel in the execution loop.
            # Returning 0.0 guarantees EV will drop below 0 for any YES bids.
            return 0.0
            
        return baseline_p_true


if __name__ == "__main__":
    sample_spot = 68171.18
    sample_date = "26FEB26"
    print(generate_viable_btc_tickers(sample_spot, sample_date))
