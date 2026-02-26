import asyncio
import aiohttp
import math
from typing import Optional, Tuple, Dict, Any

ESPN_NBA_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

class NBALiveOracle:
    """
    Live NBA Oracle that aggressively polls ESPN's hidden API to extract real-time
    score, clock, and possession data, recalculating the underlying probability 
    of the home team winning.
    """
    def __init__(self, espn_game_id: str, pre_game_p_true_home: float, base_edge: float):
        self.espn_game_id = espn_game_id
        self.pre_game_p_true_home = pre_game_p_true_home
        self.base_edge = base_edge
        
        # Live state
        self.home_score = 0
        self.away_score = 0
        self.seconds_remaining = 2880 # 48 minutes * 60
        self.possession_home: Optional[bool] = None
        self.is_paused = False

    async def poll_espn(self):
        """
        The Fast-Poll Feed: Hits the ESPN scoreboard API at a highly aggressive interval (1.5s).
        """
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.get(ESPN_NBA_SCOREBOARD_URL, timeout=5) as resp:
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
                            print(f"[NBA Oracle] ESPN rate limit (429). Backing off for {retry_after_s:.1f}s")
                            await asyncio.sleep(retry_after_s)
                            continue
                        else:
                            print(f"[NBA Oracle] Unexpected ESPN response: {resp.status}")
                except Exception as e:
                    print(f"[NBA Oracle] Error polling ESPN: {e}")
                
                await asyncio.sleep(1.5)

    def _parse_scoreboard(self, data: Dict[str, Any]):
        """
        State Extraction: Parses the ESPN response for the target game.
        """
        events = data.get("events", [])
        for event in events:
            if event.get("id") == self.espn_game_id:
                competitions = event.get("competitions", [])
                if competitions:
                    comp = competitions[0]
                    competitors = comp.get("competitors", [])
                    
                    home_team_id = None
                    away_team_id = None
                    
                    # Extract Scores & Identifiers
                    for team in competitors:
                        home_away = team.get("homeAway")
                        score = int(team.get("score", 0))
                        team_id = team.get("team", {}).get("id")
                        if home_away == "home":
                            self.home_score = score
                            home_team_id = team_id
                        else:
                            self.away_score = score
                            away_team_id = team_id
                    
                    # Extract Possession
                    situation = comp.get("situation", {})
                    possession_id = situation.get("possession")
                    if possession_id:
                        if possession_id == home_team_id:
                            self.possession_home = True
                        elif possession_id == away_team_id:
                            self.possession_home = False
                        else:
                            self.possession_home = None
                    else:
                        self.possession_home = None

                    # Extract Clock and State
                    status = event.get("status", {})
                    clock = float(status.get("clock", 0.0)) # seconds remaining in current period
                    period = int(status.get("period", 1))
                    state = status.get("type", {}).get("state", "")
                    
                    # Calculate total seconds remaining (4 quarters of 12 minutes = 720s)
                    if period <= 4:
                        quarters_left = 4 - period
                        self.seconds_remaining = quarters_left * 720 + clock
                    else:
                        # Overtime periods are 5 minutes (300 seconds)
                        # We just rely on the clock for the current OT period
                        self.seconds_remaining = clock

                    # Identify if paused (halftime, end of quarter, timeout, or play stopped)
                    if state in ["post", "halftime", "endOfPeriod"] or clock == 0.0:
                        self.is_paused = True
                    else:
                        self.is_paused = False
                break

    def calculate_live_p_true(self) -> Tuple[float, float]:
        """
        The Live-Clock Probability Adjuster:
        Returns (dynamic_p_true_home, required_edge_modifier)
        """
        total_seconds = 2880.0
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
                empirical_p = 0.5 # Going to OT
        else:
            # Baseline mathematical logic: sigmoid function where urgency scales inversely with time
            urgency = 100.0 / max(1.0, self.seconds_remaining)
            empirical_p = 1.0 / (1.0 + math.exp(-score_diff * urgency * 0.1))

        # Blend pre-game expectation with the empirical live state
        dynamic_p_home = (self.pre_game_p_true_home * decay_factor) + (empirical_p * (1.0 - decay_factor))

        # Crucial Context Modifier:
        # If the trailing team has the ball in the final 2 minutes (120 seconds),
        # bump their probability by a set margin to account for immediate scoring opportunity.
        if 0 < self.seconds_remaining <= 120:
            if score_diff < 0 and self.possession_home is True:
                dynamic_p_home += 0.05 # 5% bump for trailing home team with ball
            elif score_diff > 0 and self.possession_home is False:
                dynamic_p_home -= 0.05 # 5% bump for trailing away team with ball

        dynamic_p_home = max(0.0, min(1.0, dynamic_p_home))

        # Widen the required edge if the game is paused (clock stopped, timeouts, reviews)
        edge_modifier = 1.5 if self.is_paused else 1.0

        return dynamic_p_home, edge_modifier
