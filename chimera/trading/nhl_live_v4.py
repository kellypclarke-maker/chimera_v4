import asyncio
import aiohttp
import math
from typing import Optional, Tuple, Dict, Any

ESPN_NHL_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"

class NHLLiveOracle:
    """
    Live NHL Oracle that aggressively polls ESPN's hidden API to extract real-time
    score, clock, power play, and empty net data. Calculates the underlying probability
    and significantly adjusts required edge during extreme variance states (Empty Net).
    """
    def __init__(self, espn_game_id: str, pre_game_p_true_home: float, base_edge: float):
        self.espn_game_id = espn_game_id
        self.pre_game_p_true_home = pre_game_p_true_home
        self.base_edge = base_edge
        
        # Live state
        self.home_score = 0
        self.away_score = 0
        self.seconds_remaining = 3600 # 3 periods * 20 mins * 60
        self.is_paused = False
        
        # Hockey-Specific States
        self.home_power_play = False
        self.away_power_play = False
        self.home_empty_net = False
        self.away_empty_net = False

    async def poll_espn(self):
        """
        The Fast-Poll Feed: Hits the ESPN NHL scoreboard API at 1.5 - 2.0s intervals.
        """
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.get(ESPN_NHL_SCOREBOARD_URL, timeout=5) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            self._parse_scoreboard(data)
                        elif resp.status == 429:
                            retry_after_raw = resp.headers.get("Retry-After") or resp.headers.get("retry-after") or "2"
                            try:
                                retry_after_s = float(str(retry_after_raw).strip())
                            except (TypeError, ValueError):
                                retry_after_s = 2.0
                            retry_after_s = max(0.5, min(30.0, retry_after_s))
                            print(f"[NHL Oracle] ESPN rate limit (429). Backing off for {retry_after_s:.1f}s")
                            await asyncio.sleep(retry_after_s)
                            continue
                        else:
                            print(f"[NHL Oracle] Unexpected ESPN response: {resp.status}")
                except Exception as e:
                    print(f"[NHL Oracle] Error polling ESPN: {e}")
                
                # Polling interval: 1.5 seconds
                await asyncio.sleep(1.5)

    def _parse_scoreboard(self, data: Dict[str, Any]):
        """
        Hockey-Specific State Extraction: Parses the ESPN response for the target game.
        """
        events = data.get("events", [])
        for event in events:
            if event.get("id") == self.espn_game_id:
                competitions = event.get("competitions", [])
                if competitions:
                    comp = competitions[0]
                    competitors = comp.get("competitors", [])
                    
                    # Extract Scores & Hockey-Specific States
                    for team in competitors:
                        home_away = team.get("homeAway")
                        score = int(team.get("score", 0))
                        
                        # Power Play & Empty Net parsing
                        # Note: ESPN's payload structure can vary, but these indicators 
                        # are typically found within the competitor's object or linescores
                        linescores = team.get("linescores", [])
                        
                        # ESPN often surfaces powerPlay and goaliePulled boolean flags
                        # directly on the competitor block for NHL live games
                        is_power_play = team.get("powerPlay", False)
                        is_empty_net = team.get("goaliePulled", False)
                        
                        if home_away == "home":
                            self.home_score = score
                            self.home_power_play = is_power_play
                            self.home_empty_net = is_empty_net
                        else:
                            self.away_score = score
                            self.away_power_play = is_power_play
                            self.away_empty_net = is_empty_net

                    # Extract Clock and State
                    status = event.get("status", {})
                    clock = float(status.get("clock", 0.0)) # seconds remaining in current period
                    period = int(status.get("period", 1))
                    state = status.get("type", {}).get("state", "")
                    
                    # Calculate total seconds remaining (3 periods of 20 minutes = 1200s each)
                    if period <= 3:
                        periods_left = 3 - period
                        self.seconds_remaining = periods_left * 1200 + clock
                    else:
                        # Overtime period (usually 5 minutes = 300s in regular season)
                        self.seconds_remaining = clock

                    # Identify if paused
                    if state in ["post", "halftime", "endOfPeriod"] or clock == 0.0:
                        self.is_paused = True
                    else:
                        self.is_paused = False
                break

    def calculate_live_p_true(self) -> Tuple[float, float]:
        """
        The Discrete Event Probability Adjuster:
        Returns (dynamic_p_true_home, required_edge_modifier)
        """
        total_seconds = 3600.0
        time_fraction = max(0.0, min(1.0, self.seconds_remaining / total_seconds))
        
        # Exponential time-decay towards the current score differential
        decay_factor = time_fraction ** 1.5 
        
        score_diff = self.home_score - self.away_score
        
        # Empirical state probability
        if self.seconds_remaining <= 0:
            if score_diff > 0:
                empirical_p = 1.0
            elif score_diff < 0:
                empirical_p = 0.0
            else:
                empirical_p = 0.5 # Going to OT / Shootout
        else:
            urgency = 100.0 / max(1.0, self.seconds_remaining)
            empirical_p = 1.0 / (1.0 + math.exp(-score_diff * urgency * 0.1))

        # Blend pre-game expectation with the empirical live state
        dynamic_p_home = (self.pre_game_p_true_home * decay_factor) + (empirical_p * (1.0 - decay_factor))

        # The Power Play Modifier:
        # Bump p_true slightly for the duration of the advantage
        if self.home_power_play and not self.away_power_play:
            dynamic_p_home += 0.04
        elif self.away_power_play and not self.home_power_play:
            dynamic_p_home -= 0.04

        dynamic_p_home = max(0.0, min(1.0, dynamic_p_home))

        # The Empty Net Modifier (Extreme Variance):
        # If a team pulls their goalie, the probability of ANY goal skyrockets.
        # Order book becomes highly toxic. Widen the required edge massively to 
        # force EV negative and essentially pause trading unless the edge is absurd.
        edge_modifier = 1.0
        if self.home_empty_net or self.away_empty_net:
            edge_modifier = 10.0 # Huge modifier to guarantee we pull orders
        elif self.is_paused:
            edge_modifier = 1.5

        return dynamic_p_home, edge_modifier
