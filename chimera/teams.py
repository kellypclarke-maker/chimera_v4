from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class LeagueTeams:
    canonical: List[str]
    aliases: Mapping[str, str]

    def normalize(self, raw: str) -> Optional[str]:
        s = str(raw or "").strip().upper()
        if not s:
            return None
        if s in self.canonical:
            return s
        return self.aliases.get(s)

    def alias_candidates(self, canonical_code: str) -> List[str]:
        code = str(canonical_code or "").strip().upper()
        if not code:
            return []
        out = {code}
        # reverse lookup (canonical -> aliases)
        for a, c in self.aliases.items():
            if c == code:
                out.add(a)
        # Prefer longer tokens first to reduce ambiguous matches.
        return sorted(out, key=lambda x: (-len(x), x))


_NBA = LeagueTeams(
    canonical=[
        "ATL",
        "BOS",
        "BKN",
        "CHA",
        "CHI",
        "CLE",
        "DAL",
        "DEN",
        "DET",
        "GSW",
        "HOU",
        "IND",
        "LAC",
        "LAL",
        "MEM",
        "MIA",
        "MIL",
        "MIN",
        "NOP",
        "NYK",
        "OKC",
        "ORL",
        "PHI",
        "PHX",
        "POR",
        "SAC",
        "SAS",
        "TOR",
        "UTA",
        "WAS",
    ],
    aliases={
        "BRK": "BKN",
        "BKN": "BKN",
        "GS": "GSW",
        "NO": "NOP",
        "NOK": "NOP",
        "PHO": "PHX",
        "SA": "SAS",
        "UTAH": "UTA",
        "WSH": "WAS",
    },
)

_NHL = LeagueTeams(
    canonical=[
        "ANA",
        "BOS",
        "BUF",
        "CAR",
        "CBJ",
        "CGY",
        "CHI",
        "COL",
        "DAL",
        "DET",
        "EDM",
        "FLA",
        "LAK",
        "MIN",
        "MTL",
        "NJD",
        "NSH",
        "NYI",
        "NYR",
        "OTT",
        "PHI",
        "PIT",
        "SEA",
        "SJS",
        "STL",
        "TBL",
        "TOR",
        "UTA",
        "VAN",
        "VGK",
        "WPG",
        "WSH",
    ],
    aliases={
        "LA": "LAK",
        "L.A.": "LAK",
        "NJ": "NJD",
        "N.J.": "NJD",
        "TB": "TBL",
        "T.B.": "TBL",
        "SJ": "SJS",
        "S.J.": "SJS",
        "WSH": "WSH",
        "WAS": "WSH",
        "MON": "MTL",
        "MTL": "MTL",
        "ARZ": "UTA",  # legacy Coyotes -> Utah (Kalshi/ESPN can vary by season)
        "ARI": "UTA",
        "VGK": "VGK",
    },
)

_NFL = LeagueTeams(
    canonical=[
        "ARI",
        "ATL",
        "BAL",
        "BUF",
        "CAR",
        "CHI",
        "CIN",
        "CLE",
        "DAL",
        "DEN",
        "DET",
        "GB",
        "HOU",
        "IND",
        "JAX",
        "KC",
        "LV",
        "LAC",
        "LAR",
        "MIA",
        "MIN",
        "NE",
        "NO",
        "NYG",
        "NYJ",
        "PHI",
        "PIT",
        "SEA",
        "SF",
        "TB",
        "TEN",
        "WAS",
    ],
    aliases={
        "JAC": "JAX",
        "LA": "LAR",  # Rams (Kalshi ticker core can use LA)
        "WSH": "WAS",
    },
)


_LEAGUES: Dict[str, LeagueTeams] = {"nba": _NBA, "nhl": _NHL, "nfl": _NFL}


def normalize_team_code(raw: str, league: str) -> Optional[str]:
    lg = str(league or "").strip().lower()
    if lg not in _LEAGUES:
        return None
    return _LEAGUES[lg].normalize(raw)


def alias_candidates(canonical_code: str, league: str) -> List[str]:
    lg = str(league or "").strip().lower()
    if lg not in _LEAGUES:
        return []
    return _LEAGUES[lg].alias_candidates(canonical_code)


def canonical_teams(league: str) -> List[str]:
    lg = str(league or "").strip().lower()
    if lg not in _LEAGUES:
        return []
    return list(_LEAGUES[lg].canonical)


