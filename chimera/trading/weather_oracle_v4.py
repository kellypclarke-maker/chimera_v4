import asyncio
import aiohttp
import time
import math
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime

"""
Integration Map: Kalshi Weather Cities to NOAA Grid Coordinates
---------------------------------------------------------------
The National Weather Service API requires a specific WFO (Weather Forecast Office) 
identifier and X, Y grid coordinates for a given station.

Standard Kalshi Markets:
- NYC (Central Park): WFO = OKX, X = 33, Y = 35
- Chicago (O'Hare): WFO = LOT, X = 73, Y = 74
- Austin (Camp Mabry): WFO = EWX, X = 155, Y = 90
- Miami (MIA): WFO = MFL, X = 109, Y = 50
- Seattle (Sea-Tac): WFO = SEW, X = 125, Y = 68

Example Dict:
NOAA_STATIONS = {
    "NYC": {"wfo": "OKX", "x": 33, "y": 35},
    "CHI": {"wfo": "LOT", "x": 73, "y": 74},
    "AUS": {"wfo": "EWX", "x": 155, "y": 90},
    "MIA": {"wfo": "MFL", "x": 109, "y": 50},
    "SEA": {"wfo": "SEW", "x": 125, "y": 68},
}
"""

NOAA_STATIONS = {
    "NYC": {"wfo": "OKX", "x": 33, "y": 35},
    "CHI": {"wfo": "LOT", "x": 73, "y": 74},
    "AUS": {"wfo": "EWX", "x": 155, "y": 90},
    "MIA": {"wfo": "MFL", "x": 109, "y": 50},
    "SEA": {"wfo": "SEW", "x": 125, "y": 68},
}

class WeatherOracle:
    """
    Live Weather Oracle that aggressively polls the National Weather Service (NOAA)
    hourly forecast gridpoints, busting cache to ensure fresh data, and calculates
    dynamic volatility (sigma) based on expected rate of temperature change.
    """
    def __init__(self, station_code: str, base_sigma: float = 1.5):
        self.station_code = station_code.upper()
        if self.station_code not in NOAA_STATIONS:
            raise ValueError(f"Unknown station code: {self.station_code}")
        
        station_info = NOAA_STATIONS[self.station_code]
        self.wfo = station_info["wfo"]
        self.x = station_info["x"]
        self.y = station_info["y"]
        self.base_sigma = base_sigma
        
        self.forecast_url = f"https://api.weather.gov/gridpoints/{self.wfo}/{self.x},{self.y}/forecast/hourly"
        
        # Extracted Forecast State
        self.hourly_temperatures: List[Dict[str, Any]] = []
        self.last_fetch_time: float = 0.0

    async def poll_noaa_forecast(self, interval_seconds: float = 60.0):
        """
        The Gridpoint Poller:
        Hits the NOAA API on a loop. Includes aggressive cache busting headers.
        """
        async with aiohttp.ClientSession() as session:
            while True:
                headers = {
                    "User-Agent": "Chimera_v4_NightWatch (kellypclarke-maker@github.com)",
                    "Accept": "application/geo+json, application/json;q=0.9",
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                }
                
                try:
                    async with session.get(self.forecast_url, headers=headers, timeout=10) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            self._extract_temperatures(data)
                            self.last_fetch_time = time.time()
                        elif resp.status in [429, 503, 504]:
                            print(f"[Weather Oracle] Rate limit or unavailable from NOAA: {resp.status}")
                            await asyncio.sleep(5) # Backoff
                        else:
                            body = (await resp.text())[:180].replace("\n", " ")
                            print(f"[Weather Oracle] Unexpected NOAA response: {resp.status} | {body}")
                except Exception as e:
                    print(f"[Weather Oracle] Error polling NOAA: {e}")
                
                await asyncio.sleep(interval_seconds)

    def _extract_temperatures(self, data: Dict[str, Any]):
        """
        Parses the JSON-LD payload to extract hourly temperature forecasts.
        """
        periods = data.get("periods", [])
        if not isinstance(periods, list):
            props = data.get("properties", {})
            if isinstance(props, dict):
                nested = props.get("periods", [])
                periods = nested if isinstance(nested, list) else []
            else:
                periods = []
        extracted = []
        for period in periods:
            if not isinstance(period, dict):
                continue

            start_time_str = str(period.get("startTime") or "")
            try:
                temp = float(period.get("temperature"))
            except (TypeError, ValueError):
                continue

            # We store raw strings or basic parsed info for the calculation logic
            extracted.append({
                "start_time": start_time_str,
                "temperature": temp
            })
        
        self.hourly_temperatures = extracted

    def calculate_dynamic_sigma(self, target_window_start_iso: str, target_window_end_iso: str) -> float:
        """
        Dynamic Sigma (Volatility) Calculation:
        Extracts the forecasted temperature array for the specific time window.
        Calculates the maximum rate of change. 
        If the forecast shows high volatility (e.g. >5°F change over 2 hours), 
        dynamic_sigma is increased to widen the required edge / spread.
        """
        if not self.hourly_temperatures:
            return self.base_sigma
            
        try:
            target_start = datetime.fromisoformat(target_window_start_iso.replace('Z', '+00:00'))
            target_end = datetime.fromisoformat(target_window_end_iso.replace('Z', '+00:00'))
        except ValueError:
            # Fallback if ISO format is tricky
            return self.base_sigma

        # Filter forecast periods within the target window
        window_temps = []
        for period in self.hourly_temperatures:
            try:
                dt = datetime.fromisoformat(period["start_time"].replace('Z', '+00:00'))
                if target_start <= dt <= target_end:
                    window_temps.append(period["temperature"])
            except ValueError:
                continue
                
        if len(window_temps) < 2:
            # Not enough data points in window to determine rate of change
            return self.base_sigma

        # Calculate maximum temperature swing within the window
        max_temp = max(window_temps)
        min_temp = min(window_temps)
        max_swing = max_temp - min_temp
        
        # Base logic: If swing is > 5 degrees, we start penalizing (increasing volatility)
        # For every 1 degree of swing beyond 5, add 0.5 to our dynamic sigma
        volatility_penalty = 0.0
        swing_threshold = 5.0
        
        if max_swing > swing_threshold:
            excess_swing = max_swing - swing_threshold
            volatility_penalty = excess_swing * 0.5
            
        dynamic_sigma = self.base_sigma + volatility_penalty
        
        # Cap the sigma at a reasonable ceiling (e.g. 8.0) to avoid locking out trading entirely
        dynamic_sigma = min(8.0, dynamic_sigma)
        
        return dynamic_sigma
