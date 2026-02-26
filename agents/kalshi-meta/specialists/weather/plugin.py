from __future__ import annotations

import datetime as dt
import math
import re
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from kalshi_core.clients.fees import maker_fee_dollars
from specialists.base import CandidateProposal, RunContext
from specialists.helpers import (
    category_from_row,
    intended_maker_yes_price,
    maker_fill_prob,
    passes_maker_liquidity,
    read_cached_json,
    resolution_pointer_from_row,
    rules_hash_from_row,
    rules_pointer_from_row,
    safe_int,
    write_cached_json,
)

WEATHER_MARKET_RE = re.compile(r"^KX(?P<kind>HIGH|LOW)T(?P<code>[A-Z]+)-(?P<date>\d{2}[A-Z]{3}\d{2})")
BETWEEN_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*°?\s*to\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
BELOW_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*°?\s*(?:or below|or less|or under|or lower)", re.IGNORECASE)
ABOVE_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*°?\s*(?:or above|or more|or higher|or greater)", re.IGNORECASE)
CLI_CODE_RE = re.compile(r"\bCLI(?P<code>[A-Z]{3,4})\b", re.IGNORECASE)
RULE_STATION_RE = re.compile(r"\bK[A-Z0-9]{3,4}\b", re.IGNORECASE)
RULE_LOCATION_HINT_RE = re.compile(r'location\s+"([^"]+)"', re.IGNORECASE)

MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

# Small hardcoded map requested; keys include both raw code and prefixed variant.
LOCATION_COORDS = {
    "DC": (38.9072, -77.0369),
    "TDC": (38.9072, -77.0369),
    "SFO": (37.7749, -122.4194),
    "DEN": (39.7392, -104.9903),
    "MIAM": (25.7617, -80.1918),
}


def _parse_cli_code(text: str) -> Optional[str]:
    m = CLI_CODE_RE.search(str(text or "").upper())
    if not m:
        return None
    code = str(m.group("code") or "").strip().upper()
    if not code:
        return None
    return code


def _parse_rule_location_hint(text: str) -> str:
    m = RULE_LOCATION_HINT_RE.search(str(text or ""))
    if not m:
        return ""
    return str(m.group(1) or "").strip()


def _parse_rule_station_ids(text: str) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for m in RULE_STATION_RE.finditer(str(text or "").upper()):
        sid = str(m.group(0) or "").strip().upper()
        if not sid or sid in seen:
            continue
        seen.add(sid)
        out.append(sid)
    return out


def _cli_code_to_station_id(cli_code: str) -> Optional[str]:
    code = str(cli_code or "").strip().upper()
    if len(code) < 3:
        return None
    return "K" + code[-3:]


def _station_id_candidates_from_cli_code(cli_code: str) -> List[str]:
    code = str(cli_code or "").strip().upper()
    if len(code) < 3:
        return []
    out: List[str] = []
    seen: set[str] = set()

    def _add(x: str) -> None:
        sx = str(x or "").strip().upper()
        if not sx or sx in seen:
            return
        seen.add(sx)
        out.append(sx)

    if len(code) == 4 and code.startswith("K"):
        _add(code)
    if len(code) >= 3:
        _add("K" + code[-3:])
    if len(code) == 4 and not code.startswith("K"):
        _add("K" + code)
    _add(code)
    return out


def _station_id_from_rules(*, rules_primary: str, rules_secondary: str, title: str = "") -> Optional[str]:
    text = " ".join([str(rules_secondary or ""), str(rules_primary or ""), str(title or "")]).strip()
    station_ids = _parse_rule_station_ids(text)
    if station_ids:
        return station_ids[0]
    cli_code = _parse_cli_code(text)
    if not cli_code:
        return None
    return _cli_code_to_station_id(cli_code)


def _parse_date(token: str) -> Optional[dt.date]:
    t = str(token or "").strip().upper()
    if len(t) != 7:
        return None
    try:
        yy = int(t[:2])
        mon = MONTHS.get(t[2:5], 0)
        dd = int(t[5:7])
        if mon <= 0:
            return None
        return dt.date(2000 + yy, mon, dd)
    except Exception:
        return None


def _parse_bucket_bounds(yes_sub_title: str) -> Optional[Tuple[Optional[float], Optional[float]]]:
    s = str(yes_sub_title or "").strip()
    if not s:
        return None
    m = BETWEEN_RE.search(s)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        return (min(a, b), max(a, b))
    m = BELOW_RE.search(s)
    if m:
        return (None, float(m.group(1)))
    m = ABOVE_RE.search(s)
    if m:
        return (float(m.group(1)), None)
    return None


def _bucket_kind(*, lower: Optional[float], upper: Optional[float]) -> str:
    if lower is None and upper is not None:
        return "lt"
    if lower is not None and upper is None:
        return "gt"
    if lower is not None and upper is not None:
        return "between"
    return ""


def _infer_comparator_hints(text: str) -> set[str]:
    s = str(text or "").lower()
    hints: set[str] = set()
    if re.search(r"\bbetween\b", s) or re.search(r"\d+\s*-\s*\d+", s):
        hints.add("between")
    if re.search(r"\bless than\b|\bbelow\b|\bunder\b|<", s):
        hints.add("lt")
    if re.search(r"\bgreater than\b|\babove\b|\bover\b|>", s):
        hints.add("gt")
    return hints


def _bucket_matches_comparator(*, bucket_kind: str, hints: set[str]) -> bool:
    if not hints:
        return True
    return str(bucket_kind) in hints


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0.0:
        return 0.0 if x < mu else 1.0
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def _bucket_probability(*, lower: Optional[float], upper: Optional[float], mu: float, sigma: float) -> float:
    if lower is not None and upper is not None:
        lo = min(lower, upper) - 0.5
        hi = max(lower, upper) + 0.5
        return max(0.0, min(1.0, _normal_cdf(hi, mu, sigma) - _normal_cdf(lo, mu, sigma)))
    if upper is not None:
        return max(0.0, min(1.0, _normal_cdf(upper + 0.5, mu, sigma)))
    if lower is not None:
        return max(0.0, min(1.0, 1.0 - _normal_cdf(lower - 0.5, mu, sigma)))
    return float("nan")


def _iso_to_date(x: str) -> Optional[dt.date]:
    s = str(x or "").strip()
    if not s:
        return None
    try:
        ts = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        return ts.date()
    except Exception:
        return None


def _iso_to_datetime(x: str) -> Optional[dt.datetime]:
    s = str(x or "").strip()
    if not s:
        return None
    try:
        t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        return t
    except Exception:
        return None


def _f_from_observation_value(*, value: object, unit_code: str) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    u = str(unit_code or "").strip().lower()
    if "degc" in u or u.endswith(":c"):
        return (v * 9.0 / 5.0) + 32.0
    if "degf" in u or u.endswith(":f"):
        return float(v)
    return None


def _resolve_local_tz(
    *,
    target_date: dt.date,
    hourly_payload: Optional[Dict[str, object]],
    forecast_payload: Dict[str, object],
) -> Optional[dt.tzinfo]:
    payloads: List[Optional[Dict[str, object]]] = [hourly_payload, forecast_payload]
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        props = payload.get("properties")
        if not isinstance(props, dict):
            continue
        periods = props.get("periods")
        if not isinstance(periods, list):
            continue
        fallback_tz: Optional[dt.tzinfo] = None
        for p in periods:
            if not isinstance(p, dict):
                continue
            start_dt = _iso_to_datetime(str(p.get("startTime") or ""))
            if start_dt is None:
                continue
            if fallback_tz is None:
                fallback_tz = start_dt.tzinfo
            if start_dt.date() == target_date:
                return start_dt.tzinfo
        if fallback_tz is not None:
            return fallback_tz
    return None


def _hourly_temps_for_date(*, hourly_payload: Optional[Dict[str, object]], target_date: dt.date) -> List[float]:
    if not isinstance(hourly_payload, dict):
        return []
    h_props = hourly_payload.get("properties")
    if not isinstance(h_props, dict):
        return []
    h_periods = h_props.get("periods")
    if not isinstance(h_periods, list):
        return []
    temps: List[float] = []
    for p in h_periods:
        if not isinstance(p, dict):
            continue
        if str(p.get("temperatureUnit") or "").strip().upper() != "F":
            continue
        start_dt = _iso_to_datetime(str(p.get("startTime") or ""))
        if start_dt is None or start_dt.date() != target_date:
            continue
        try:
            temp = float(p.get("temperature"))
        except Exception:
            continue
        temps.append(temp)
    return temps


def _observation_temps_for_date(
    *,
    observed_payload: Optional[Dict[str, object]],
    target_date: dt.date,
    local_tz: Optional[dt.tzinfo],
) -> List[float]:
    if not isinstance(observed_payload, dict):
        return []
    features = observed_payload.get("features")
    if not isinstance(features, list):
        return []
    temps: List[float] = []
    for feat in features:
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties")
        if not isinstance(props, dict):
            continue
        ts = _iso_to_datetime(str(props.get("timestamp") or ""))
        if ts is None:
            continue
        if local_tz is not None:
            try:
                local_date = ts.astimezone(local_tz).date()
            except Exception:
                local_date = ts.date()
        else:
            local_date = ts.date()
        if local_date != target_date:
            continue
        temp_obj = props.get("temperature")
        if not isinstance(temp_obj, dict):
            continue
        value = temp_obj.get("value")
        if value is None:
            continue
        temp_f = _f_from_observation_value(value=value, unit_code=str(temp_obj.get("unitCode") or ""))
        if temp_f is None:
            continue
        temps.append(float(temp_f))
    return temps


def _latest_observation_timestamp(observed_payload: Optional[Dict[str, object]]) -> str:
    if not isinstance(observed_payload, dict):
        return ""
    features = observed_payload.get("features")
    if not isinstance(features, list):
        return ""
    best: Optional[dt.datetime] = None
    for feat in features:
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties")
        if not isinstance(props, dict):
            continue
        ts = _iso_to_datetime(str(props.get("timestamp") or ""))
        if ts is None:
            continue
        if best is None or ts > best:
            best = ts
    return best.isoformat() if best is not None else ""


def _forecast_mu_for_date(
    *,
    forecast_payload: Dict[str, object],
    target_date: dt.date,
    kind: str,
    hourly_payload: Optional[Dict[str, object]] = None,
    observed_payload: Optional[Dict[str, object]] = None,
) -> Optional[float]:
    kind_norm = str(kind).upper()
    local_tz = _resolve_local_tz(target_date=target_date, hourly_payload=hourly_payload, forecast_payload=forecast_payload)
    hourly_temps = _hourly_temps_for_date(hourly_payload=hourly_payload, target_date=target_date)
    observed_temps = _observation_temps_for_date(
        observed_payload=observed_payload,
        target_date=target_date,
        local_tz=local_tz,
    )
    temps = [float(x) for x in (observed_temps + hourly_temps)]

    if kind_norm == "LOW":
        if temps:
            return float(min(temps))
        return None
    if kind_norm == "HIGH":
        if temps:
            return float(max(temps))

    props = forecast_payload.get("properties")
    if not isinstance(props, dict):
        return None
    periods = props.get("periods")
    if not isinstance(periods, list):
        return None

    highs_primary: List[float] = []
    highs_fallback: List[float] = []
    for p in periods:
        if not isinstance(p, dict):
            continue
        if str(p.get("temperatureUnit") or "").strip().upper() != "F":
            continue
        start_dt = _iso_to_datetime(str(p.get("startTime") or ""))
        end_dt = _iso_to_datetime(str(p.get("endTime") or ""))
        start_date = start_dt.date() if start_dt is not None else None
        end_date = end_dt.date() if end_dt is not None else None
        is_daytime = p.get("isDaytime")
        try:
            temp = float(p.get("temperature"))
        except Exception:
            continue
        if kind_norm == "HIGH":
            if is_daytime is True and start_date == target_date:
                highs_primary.append(temp)
            if start_date == target_date:
                highs_fallback.append(temp)

    if kind_norm == "HIGH":
        if highs_primary:
            return float(max(highs_primary))
        if highs_fallback:
            return float(max(highs_fallback))
        return None
    return None


def _nws_headers(user_agent: str) -> Dict[str, str]:
    _ = user_agent
    ua = "Chimera_v4_NightWatch (kellypclarke-maker@github.com)"
    return {
        "User-Agent": ua,
        "Accept": "application/geo+json",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }


def _station_cache_key(station_id: str) -> str:
    return "".join(ch for ch in str(station_id or "").strip().upper() if ch.isalnum())


def _stable_text_key(text: str) -> str:
    return "".join(ch for ch in str(text or "").strip().upper() if ch.isalnum())[:64]


def _station_cache_path(*, cache_dir: Path, station_id: str) -> Path:
    return cache_dir / "stations" / f"{_station_cache_key(station_id)}.json"


def _station_resolution_cache_path(*, cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / "station_resolution" / f"{_stable_text_key(cache_key)}.json"


def _sigma_profile_cache_path(*, cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / "sigma_profiles" / f"{_stable_text_key(cache_key)}.json"


def _cli_locations_cache_path(*, cache_dir: Path) -> Path:
    return cache_dir / "cli_locations.json"


@dataclass(frozen=True)
class WeatherStationContext:
    station_id: str
    station_name: str
    station_lat: float
    station_lon: float
    cli_code: str
    location_hint: str
    source: str


@dataclass(frozen=True)
class ProviderSignal:
    provider: str
    mu_high_f: Optional[float]
    mu_low_f: Optional[float]
    as_of_ts: Optional[dt.datetime]
    metadata: Dict[str, object]


def _extract_cli_locations(payload: Dict[str, object]) -> set[str]:
    loc = payload.get("locations")
    if not isinstance(loc, dict):
        return set()
    out: set[str] = set()
    for k in loc.keys():
        sk = str(k or "").strip().upper()
        if sk:
            out.add(sk)
    return out


def _fetch_cli_locations(
    *,
    cache_dir: Path,
    ttl_minutes: float,
    user_agent: str,
) -> set[str]:
    cache_path = _cli_locations_cache_path(cache_dir=cache_dir)
    cached = read_cached_json(cache_path, ttl_minutes=ttl_minutes)
    if isinstance(cached, dict):
        got = _extract_cli_locations(cached)
        if got:
            return got

    headers = _nws_headers(user_agent)
    try:
        with requests.Session() as s:
            resp = s.get("https://api.weather.gov/products/types/CLI/locations", headers=headers, timeout=20.0)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                return set()
            write_cached_json(cache_path, data)
            return _extract_cli_locations(data)
    except Exception:
        return set()


def _station_name_score(*, station_name: str, location_hint: str) -> float:
    sn = str(station_name or "").strip().lower()
    lh = str(location_hint or "").strip().lower()
    if not sn or not lh:
        return 0.0
    if sn == lh:
        return 1.0
    if sn in lh or lh in sn:
        return 0.8
    station_tokens = {t for t in re.split(r"[^a-z0-9]+", sn) if t}
    hint_tokens = {t for t in re.split(r"[^a-z0-9]+", lh) if t}
    if not station_tokens or not hint_tokens:
        return 0.0
    overlap = station_tokens.intersection(hint_tokens)
    if not overlap:
        return 0.0
    return float(len(overlap)) / float(max(len(hint_tokens), 1))


def _score_station_candidate(
    *,
    source: str,
    station_name: str,
    location_hint: str,
    cli_code: str,
    station_id: str,
    cli_locations: set[str],
) -> float:
    base = 0.0
    src = str(source or "")
    if src == "rules_station_id":
        base += 3.0
    elif src == "cli_code":
        base += 2.0
    elif src == "ticker_code_fallback":
        base += 1.0

    if cli_code and cli_code in cli_locations:
        expected = _cli_code_to_station_id(cli_code)
        if expected and str(expected).upper() == str(station_id).upper():
            base += 0.75
    base += _station_name_score(station_name=station_name, location_hint=location_hint)
    return float(base)


def _parse_forecast_timestamp(
    *,
    forecast_payload: Optional[Dict[str, object]],
    forecast_hourly_payload: Optional[Dict[str, object]],
    observations_payload: Optional[Dict[str, object]],
) -> Optional[dt.datetime]:
    fields: List[str] = []
    props = forecast_payload.get("properties") if isinstance(forecast_payload, dict) else {}
    hprops = forecast_hourly_payload.get("properties") if isinstance(forecast_hourly_payload, dict) else {}
    if isinstance(props, dict):
        fields.extend([str(props.get("updated") or "").strip(), str(props.get("generatedAt") or "").strip()])
    if isinstance(hprops, dict):
        fields.extend([str(hprops.get("updated") or "").strip(), str(hprops.get("generatedAt") or "").strip()])
    obs_ts = _latest_observation_timestamp(observations_payload)
    if obs_ts:
        fields.append(obs_ts)
    best: Optional[dt.datetime] = None
    for f in fields:
        t = _iso_to_datetime(f)
        if t is None:
            continue
        if best is None or t > best:
            best = t
    return best


def _season_from_month(month: int) -> str:
    m = int(month)
    if m in {12, 1, 2}:
        return "DJF"
    if m in {3, 4, 5}:
        return "MAM"
    if m in {6, 7, 8}:
        return "JJA"
    return "SON"


def _parse_float_map(raw: object) -> Dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in raw.items():
        key = str(k or "").strip().upper()
        if not key:
            continue
        try:
            out[key] = float(v)
        except Exception:
            continue
    return out


def _load_sigma_profile(
    *,
    config: Dict[str, object],
    cache_dir: Path,
    ttl_minutes: float,
) -> Dict[str, object]:
    path_raw = str(config.get("weather_sigma_profile_path") or "").strip()
    if not path_raw:
        return {}

    p = Path(path_raw)
    if not p.is_absolute():
        p = Path.cwd() / p
    cache_key = str(p.resolve()) if p.exists() else str(p)
    cache_path = _sigma_profile_cache_path(cache_dir=cache_dir, cache_key=cache_key)
    cached = read_cached_json(cache_path, ttl_minutes=ttl_minutes)
    if isinstance(cached, dict):
        return cached

    if not p.exists() or not p.is_file():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            write_cached_json(cache_path, payload)
            return payload
    except Exception:
        return {}
    return {}


def _effective_weather_sigma(
    *,
    kind: str,
    code: str,
    station_id: str,
    target_date: dt.date,
    local_tz: Optional[dt.tzinfo],
    as_of_ts: Optional[dt.datetime],
    config: Dict[str, object],
    cache_dir: Path,
    cache_ttl_minutes: float,
) -> Tuple[float, Dict[str, object]]:
    kind_norm = str(kind or "").strip().upper()
    code_norm = str(code or "").strip().upper()
    station_norm = str(station_id or "").strip().upper()
    base = float(config.get("weather_sigma_high_f", 3.0) if kind_norm == "HIGH" else config.get("weather_sigma_low_f", 3.5))

    lead_anchor_hour = int(config.get("weather_sigma_high_anchor_hour_local", 15) if kind_norm == "HIGH" else config.get("weather_sigma_low_anchor_hour_local", 6))
    step_hours = max(1.0, float(config.get("weather_sigma_lead_step_hours", 12.0)))
    lead_step_mult = float(config.get("weather_sigma_lead_step_mult", 0.04))
    lead_max_steps = max(0.0, float(config.get("weather_sigma_lead_max_steps", 8.0)))
    sigma_floor = max(0.25, float(config.get("weather_sigma_floor_f", 1.5)))
    sigma_cap = max(sigma_floor, float(config.get("weather_sigma_cap_f", 8.0)))

    season = _season_from_month(target_date.month)
    season_mults = {
        "DJF": float(config.get("weather_sigma_season_mult_djf", 1.05)),
        "MAM": float(config.get("weather_sigma_season_mult_mam", 1.00)),
        "JJA": float(config.get("weather_sigma_season_mult_jja", 0.95)),
        "SON": float(config.get("weather_sigma_season_mult_son", 1.00)),
    }
    season_mult = float(season_mults.get(season, 1.0))

    station_mults = _parse_float_map(config.get("weather_sigma_station_mults"))
    code_mults = _parse_float_map(config.get("weather_sigma_code_mults"))
    station_mult = float(station_mults.get(station_norm, code_mults.get(code_norm, 1.0)))

    profile = _load_sigma_profile(config=config, cache_dir=cache_dir, ttl_minutes=cache_ttl_minutes)
    profile_station_mult = 1.0
    profile_season_mult = 1.0
    profile_lead_step_mult = lead_step_mult
    profile_lead_bucket_mult = 1.0
    profile_lead_buckets: List[Dict[str, float]] = []
    if isinstance(profile, dict):
        stations = profile.get("stations")
        if isinstance(stations, dict):
            st = stations.get(station_norm)
            if isinstance(st, dict):
                k = st.get("high") if kind_norm == "HIGH" else st.get("low")
                if isinstance(k, dict):
                    try:
                        profile_station_mult = float(k.get("mult", 1.0))
                    except Exception:
                        profile_station_mult = 1.0
                    sm = _parse_float_map(k.get("season_mult"))
                    if sm:
                        profile_season_mult = float(sm.get(season, 1.0))
                    try:
                        profile_lead_step_mult = float(k.get("lead_step_mult", profile_lead_step_mult))
                    except Exception:
                        pass
                    raw_buckets = k.get("lead_buckets")
                    if isinstance(raw_buckets, list):
                        for b in raw_buckets:
                            if not isinstance(b, dict):
                                continue
                            max_h = _safe_float(b.get("max_h"))
                            mult = _safe_float(b.get("mult"))
                            if max_h is None or mult is None or max_h < 0:
                                continue
                            profile_lead_buckets.append({"max_h": float(max_h), "mult": float(mult)})

    lead_hours = 0.0
    if as_of_ts is not None:
        tzinfo = local_tz or dt.timezone.utc
        anchor_local = dt.datetime.combine(target_date, dt.time(hour=min(max(int(lead_anchor_hour), 0), 23), tzinfo=tzinfo))
        try:
            anchor_utc = anchor_local.astimezone(dt.timezone.utc)
        except Exception:
            anchor_utc = anchor_local.replace(tzinfo=dt.timezone.utc)
        lead_hours = max(0.0, (anchor_utc - as_of_ts.astimezone(dt.timezone.utc)).total_seconds() / 3600.0)

    lead_steps = min(lead_max_steps, max(0.0, lead_hours / step_hours))
    lead_mult = 1.0 + (lead_steps * profile_lead_step_mult)
    if profile_lead_buckets:
        for b in sorted(profile_lead_buckets, key=lambda x: float(x.get("max_h", 0.0))):
            if lead_hours <= float(b.get("max_h", 0.0)):
                profile_lead_bucket_mult = float(b.get("mult", 1.0))
                break

    sigma_raw = (
        float(base)
        * float(lead_mult)
        * float(season_mult)
        * float(station_mult)
        * float(profile_station_mult)
        * float(profile_season_mult)
        * float(profile_lead_bucket_mult)
    )
    sigma = min(sigma_cap, max(sigma_floor, sigma_raw))
    meta = {
        "base_sigma": float(base),
        "lead_hours": float(lead_hours),
        "lead_steps": float(lead_steps),
        "lead_mult": float(lead_mult),
        "season": season,
        "season_mult": float(season_mult),
        "station_mult": float(station_mult),
        "profile_station_mult": float(profile_station_mult),
        "profile_season_mult": float(profile_season_mult),
        "profile_lead_bucket_mult": float(profile_lead_bucket_mult),
        "sigma_raw": float(sigma_raw),
        "sigma_floor": float(sigma_floor),
        "sigma_cap": float(sigma_cap),
    }
    return (float(sigma), meta)

def _extract_station_coords(station_payload: Dict[str, object]) -> Optional[Tuple[float, float]]:
    geom = station_payload.get("geometry")
    if not isinstance(geom, dict):
        return None
    coords = geom.get("coordinates")
    if not (isinstance(coords, list) and len(coords) >= 2):
        return None
    try:
        lon = float(coords[0])
        lat = float(coords[1])
    except Exception:
        return None
    return (lat, lon)


def _extract_station_name(station_payload: Dict[str, object]) -> str:
    props = station_payload.get("properties")
    if not isinstance(props, dict):
        return ""
    return str(props.get("name") or props.get("stationIdentifier") or "").strip()


def _fetch_station_geojson(*, station_id: str, cache_dir: Path, ttl_minutes: float, user_agent: str) -> Optional[Dict[str, object]]:
    cache_path = _station_cache_path(cache_dir=cache_dir, station_id=station_id)
    cached = read_cached_json(cache_path, ttl_minutes=ttl_minutes)
    if cached is not None:
        return cached
    sid = str(station_id or "").strip().upper()
    if not sid:
        return None

    try:
        with requests.Session() as s:
            headers = _nws_headers(user_agent)
            resp = s.get(f"https://api.weather.gov/stations/{sid}", headers=headers, timeout=20.0)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    write_cached_json(cache_path, data)
                    return data
            # Fallback endpoint; resolves IDs that may not be available at /stations/{id}.
            q_resp = s.get("https://api.weather.gov/stations", params={"id": sid}, headers=headers, timeout=20.0)
            q_resp.raise_for_status()
            q_data = q_resp.json()
            if not isinstance(q_data, dict):
                return None
            features = q_data.get("features")
            if not isinstance(features, list) or not features:
                return None
            feat = features[0]
            if not isinstance(feat, dict):
                return None
            write_cached_json(cache_path, feat)
            return feat
    except Exception:
        return None


def _resolve_station_context(
    *,
    rules_primary: str,
    rules_secondary: str,
    title: str,
    ticker_code: str,
    cache_dir: Path,
    ttl_minutes: float,
    user_agent: str,
) -> Optional[WeatherStationContext]:
    rules_blob = " ".join([str(rules_secondary or ""), str(rules_primary or ""), str(title or "")]).strip()
    cli_code = _parse_cli_code(rules_blob) or ""
    location_hint = _parse_rule_location_hint(rules_blob)
    explicit_station_ids = _parse_rule_station_ids(rules_blob)

    cache_key = "|".join(
        [
            str(cli_code or ""),
            str(ticker_code or "").strip().upper(),
            str(location_hint or "").strip().lower(),
            ",".join(explicit_station_ids),
        ]
    )
    cache_path = _station_resolution_cache_path(cache_dir=cache_dir, cache_key=cache_key)
    cached = read_cached_json(cache_path, ttl_minutes=ttl_minutes)
    if isinstance(cached, dict):
        try:
            sid = str(cached.get("station_id") or "").strip().upper()
            sname = str(cached.get("station_name") or "").strip()
            slat = float(cached.get("station_lat"))
            slon = float(cached.get("station_lon"))
            source = str(cached.get("source") or "").strip()
            ccode = str(cached.get("cli_code") or "").strip().upper()
            hint = str(cached.get("location_hint") or "").strip()
            if sid and sname:
                return WeatherStationContext(
                    station_id=sid,
                    station_name=sname,
                    station_lat=slat,
                    station_lon=slon,
                    cli_code=ccode,
                    location_hint=hint,
                    source=source or "cache",
                )
        except Exception:
            pass

    cli_locations = _fetch_cli_locations(cache_dir=cache_dir, ttl_minutes=ttl_minutes, user_agent=user_agent)
    candidates: List[Tuple[str, str]] = []
    seen: set[str] = set()

    def _add(source: str, station_id: str) -> None:
        sid = str(station_id or "").strip().upper()
        if not sid or sid in seen:
            return
        seen.add(sid)
        candidates.append((source, sid))

    for sid in explicit_station_ids:
        _add("rules_station_id", sid)

    if cli_code:
        for sid in _station_id_candidates_from_cli_code(cli_code):
            _add("cli_code", sid)

    tcode = str(ticker_code or "").strip().upper()
    if tcode:
        for sid in _station_id_candidates_from_cli_code(tcode):
            _add("ticker_code_fallback", sid)

    best: Optional[Tuple[float, WeatherStationContext]] = None
    for source, sid in candidates:
        station_geo = _fetch_station_geojson(
            station_id=sid,
            cache_dir=cache_dir,
            ttl_minutes=ttl_minutes,
            user_agent=user_agent,
        )
        if station_geo is None:
            continue
        coords = _extract_station_coords(station_geo)
        if coords is None:
            continue
        station_name = _extract_station_name(station_geo)
        if not station_name:
            continue
        score = _score_station_candidate(
            source=source,
            station_name=station_name,
            location_hint=location_hint,
            cli_code=cli_code,
            station_id=sid,
            cli_locations=cli_locations,
        )
        context = WeatherStationContext(
            station_id=sid,
            station_name=station_name,
            station_lat=float(coords[0]),
            station_lon=float(coords[1]),
            cli_code=cli_code,
            location_hint=location_hint,
            source=source,
        )
        if best is None or score > best[0]:
            best = (score, context)

    if best is None:
        return None
    out = best[1]
    write_cached_json(
        cache_path,
        {
            "station_id": out.station_id,
            "station_name": out.station_name,
            "station_lat": out.station_lat,
            "station_lon": out.station_lon,
            "cli_code": out.cli_code,
            "location_hint": out.location_hint,
            "source": out.source,
        },
    )
    return out


def _fetch_nws_forecast(*, code: str, lat: float, lon: float, cache_dir: Path, ttl_minutes: float, user_agent: str) -> Optional[Dict[str, object]]:
    key = _station_cache_key(code)
    points_cache = cache_dir / f"points_{key}.json"
    forecast_cache = cache_dir / f"forecast_{key}.json"
    points_data = read_cached_json(points_cache, ttl_minutes=ttl_minutes)
    forecast_data = read_cached_json(forecast_cache, ttl_minutes=ttl_minutes)
    if forecast_data is not None:
        return forecast_data

    headers = _nws_headers(user_agent)
    try:
        with requests.Session() as s:
            if points_data is None:
                points_url = f"https://api.weather.gov/points/{lat},{lon}"
                p_resp = s.get(points_url, headers=headers, timeout=20.0)
                p_resp.raise_for_status()
                p_json = p_resp.json()
                if not isinstance(p_json, dict):
                    return None
                points_data = p_json
                write_cached_json(points_cache, points_data)

            props = points_data.get("properties") if isinstance(points_data, dict) else None
            if not isinstance(props, dict):
                return None
            forecast_url = str(props.get("forecast") or "").strip()
            if not forecast_url:
                return None
            f_resp = s.get(forecast_url, headers=headers, timeout=20.0)
            f_resp.raise_for_status()
            f_json = f_resp.json()
            if not isinstance(f_json, dict):
                return None
            write_cached_json(forecast_cache, f_json)
            return f_json
    except Exception:
        return None


def _fetch_nws_hourly_forecast(
    *,
    code: str,
    lat: float,
    lon: float,
    cache_dir: Path,
    ttl_minutes: float,
    user_agent: str,
) -> Optional[Dict[str, object]]:
    key = _station_cache_key(code)
    points_cache = cache_dir / f"points_{key}.json"
    hourly_cache = cache_dir / f"forecast_hourly_{key}.json"
    points_data = read_cached_json(points_cache, ttl_minutes=ttl_minutes)
    hourly_data = read_cached_json(hourly_cache, ttl_minutes=ttl_minutes)
    if hourly_data is not None:
        return hourly_data

    headers = _nws_headers(user_agent)
    try:
        with requests.Session() as s:
            if points_data is None:
                points_url = f"https://api.weather.gov/points/{lat},{lon}"
                p_resp = s.get(points_url, headers=headers, timeout=20.0)
                p_resp.raise_for_status()
                p_json = p_resp.json()
                if not isinstance(p_json, dict):
                    return None
                points_data = p_json
                write_cached_json(points_cache, points_data)

            props = points_data.get("properties") if isinstance(points_data, dict) else None
            if not isinstance(props, dict):
                return None
            hourly_url = str(props.get("forecastHourly") or "").strip()
            if not hourly_url:
                return None
            h_resp = s.get(hourly_url, headers=headers, timeout=20.0)
            h_resp.raise_for_status()
            h_json = h_resp.json()
            if not isinstance(h_json, dict):
                return None
            write_cached_json(hourly_cache, h_json)
            return h_json
    except Exception:
        return None


def _fetch_station_observations(
    *,
    station_id: str,
    cache_dir: Path,
    ttl_minutes: float,
    user_agent: str,
) -> Optional[Dict[str, object]]:
    key = _station_cache_key(station_id)
    obs_cache = cache_dir / f"observations_{key}.json"
    cached = read_cached_json(obs_cache, ttl_minutes=ttl_minutes)
    if cached is not None:
        return cached

    sid = str(station_id or "").strip().upper()
    if not sid:
        return None

    headers = _nws_headers(user_agent)
    try:
        with requests.Session() as s:
            url = f"https://api.weather.gov/stations/{sid}/observations?limit=300"
            resp = s.get(url, headers=headers, timeout=20.0)
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, dict):
                return None
            write_cached_json(obs_cache, payload)
            return payload
    except Exception:
        return None


def _safe_float(raw: object) -> Optional[float]:
    try:
        v = float(raw)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _provider_mu_for_kind(*, signal: ProviderSignal, kind: str) -> Optional[float]:
    k = str(kind or "").strip().upper()
    if k == "HIGH":
        return _safe_float(signal.mu_high_f)
    if k == "LOW":
        return _safe_float(signal.mu_low_f)
    return None


def _parse_weight_map(raw: object) -> Dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in raw.items():
        key = str(k or "").strip().lower()
        if not key:
            continue
        fv = _safe_float(v)
        if fv is None or fv <= 0:
            continue
        out[key] = float(fv)
    return out


def _weighted_ensemble_mu(
    *,
    signals: Sequence[ProviderSignal],
    kind: str,
    weights: Dict[str, float],
) -> Tuple[Optional[float], Dict[str, object]]:
    weighted_sum = 0.0
    weight_sum = 0.0
    used: List[Tuple[str, float, float]] = []
    mus: List[float] = []
    as_of_ts: Optional[dt.datetime] = None

    for s in signals:
        mu = _provider_mu_for_kind(signal=s, kind=kind)
        if mu is None:
            continue
        w = float(weights.get(str(s.provider).strip().lower(), 0.0))
        if w <= 0.0:
            continue
        weighted_sum += float(mu) * float(w)
        weight_sum += float(w)
        used.append((s.provider, float(mu), float(w)))
        mus.append(float(mu))
        if s.as_of_ts is not None and (as_of_ts is None or s.as_of_ts > as_of_ts):
            as_of_ts = s.as_of_ts

    if weight_sum <= 0.0:
        return (None, {"providers": [], "weights": {}, "disagreement_f": 0.0, "as_of_ts": ""})

    mu = weighted_sum / weight_sum
    disagreement = (max(mus) - min(mus)) if mus else 0.0
    providers = [u[0] for u in used]
    norm_weights = {u[0]: (u[2] / weight_sum) for u in used}
    return (
        float(mu),
        {
            "providers": providers,
            "weights": norm_weights,
            "disagreement_f": float(disagreement),
            "as_of_ts": as_of_ts.isoformat() if as_of_ts is not None else "",
        },
    )


def _deterministic_bucket_probability_from_extrema(
    *,
    kind: str,
    lower: Optional[float],
    upper: Optional[float],
    observed_temps_f: Sequence[float],
) -> Optional[float]:
    if not observed_temps_f:
        return None
    k = str(kind or "").strip().upper()
    obs_min = min(float(x) for x in observed_temps_f)
    obs_max = max(float(x) for x in observed_temps_f)
    bucket = _bucket_kind(lower=lower, upper=upper)

    if k == "HIGH":
        if upper is not None and obs_max > float(upper):
            if bucket in {"lt", "between"}:
                return 0.0
        if lower is not None and obs_max >= float(lower):
            if bucket == "gt":
                return 1.0
    elif k == "LOW":
        if lower is not None and obs_min < float(lower):
            if bucket in {"gt", "between"}:
                return 0.0
        if upper is not None and obs_min <= float(upper):
            if bucket == "lt":
                return 1.0
    return None


def _calibration_cache_path(*, cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / "calibration_profiles" / f"{_stable_text_key(cache_key)}.json"


def _load_weather_calibration_profile(
    *,
    config: Dict[str, object],
    cache_dir: Path,
    ttl_minutes: float,
) -> Dict[str, object]:
    path_raw = str(config.get("weather_calibration_profile_path") or "").strip()
    if not path_raw:
        return {}

    p = Path(path_raw)
    if not p.is_absolute():
        p = Path.cwd() / p
    cache_key = str(p.resolve()) if p.exists() else str(p)
    cache_path = _calibration_cache_path(cache_dir=cache_dir, cache_key=cache_key)
    cached = read_cached_json(cache_path, ttl_minutes=ttl_minutes)
    if isinstance(cached, dict):
        return cached

    if not p.exists() or not p.is_file():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    write_cached_json(cache_path, payload)
    return payload


def _apply_isotonic_points(*, p: float, points: Sequence[Sequence[object]]) -> float:
    if not points:
        return _clamp01(p)
    parsed: List[Tuple[float, float]] = []
    for pt in points:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        x = _safe_float(pt[0])
        y = _safe_float(pt[1])
        if x is None or y is None:
            continue
        parsed.append((float(x), _clamp01(float(y))))
    if not parsed:
        return _clamp01(p)
    parsed.sort(key=lambda t: t[0])
    x = float(p)
    if x <= parsed[0][0]:
        return parsed[0][1]
    if x >= parsed[-1][0]:
        return parsed[-1][1]
    for i in range(1, len(parsed)):
        x0, y0 = parsed[i - 1]
        x1, y1 = parsed[i]
        if x0 <= x <= x1:
            if x1 <= x0:
                return y1
            frac = (x - x0) / (x1 - x0)
            return _clamp01(y0 + frac * (y1 - y0))
    return _clamp01(p)


def _apply_calibration_method(*, p: float, method: str, segment: Dict[str, object]) -> float:
    m = str(method or "").strip().lower()
    base = _clamp01(float(p))
    if m == "platt":
        a = _safe_float(segment.get("a"))
        b = _safe_float(segment.get("b"))
        if a is None or b is None:
            return base
        x = max(1e-6, min(1.0 - 1e-6, base))
        logit = math.log(x / (1.0 - x))
        z = (float(a) * logit) + float(b)
        return _clamp01(1.0 / (1.0 + math.exp(-z)))
    if m == "isotonic":
        pts = segment.get("points")
        if isinstance(pts, list):
            return _apply_isotonic_points(p=base, points=pts)
    return base


def _calibrate_weather_probability(
    *,
    p_true_raw: float,
    kind: str,
    station_id: str,
    lead_hours: float,
    profile: Dict[str, object],
) -> Tuple[float, Dict[str, str]]:
    segments = profile.get("segments") if isinstance(profile, dict) else None
    if not isinstance(segments, list):
        return (_clamp01(p_true_raw), {"segment": "", "method": ""})

    k = str(kind or "").strip().upper()
    sid = str(station_id or "").strip().upper()
    lead = max(0.0, float(lead_hours))

    best: Optional[Tuple[float, Dict[str, object]]] = None
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        score = 0.0
        seg_kind = str(seg.get("kind") or "").strip().upper()
        if seg_kind:
            if seg_kind != k:
                continue
            score += 2.0
        seg_station = str(seg.get("station_id") or "").strip().upper()
        if seg_station:
            if seg_station != sid:
                continue
            score += 4.0
        lead_min = _safe_float(seg.get("lead_h_min"))
        lead_max = _safe_float(seg.get("lead_h_max"))
        if lead_min is not None and lead < float(lead_min):
            continue
        if lead_max is not None and lead > float(lead_max):
            continue
        if lead_min is not None or lead_max is not None:
            score += 1.0
        sample_min = _safe_float(seg.get("min_samples"))
        if sample_min is not None:
            seg_samples = _safe_float(seg.get("samples")) or 0.0
            if seg_samples < float(sample_min):
                continue
        if best is None or score > best[0]:
            best = (score, seg)

    if best is None:
        return (_clamp01(p_true_raw), {"segment": "", "method": ""})

    seg = best[1]
    seg_id = str(seg.get("id") or "")
    method = str(seg.get("method") or "").strip().lower()
    p_cal = _apply_calibration_method(p=float(p_true_raw), method=method, segment=seg)
    return (_clamp01(p_cal), {"segment": seg_id, "method": method})


def _provider_cache_path(*, cache_dir: Path, provider: str, cache_key: str) -> Path:
    safe_provider = _stable_text_key(provider) or "provider"
    safe_key = _stable_text_key(cache_key) or "key"
    return cache_dir / "providers" / safe_provider / f"{safe_key}.json"


def _iso_date_part(x: object) -> str:
    s = str(x or "").strip()
    if not s:
        return ""
    t = _iso_to_datetime(s)
    if t is not None:
        return t.date().isoformat()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    return ""


def _build_nws_signal(
    *,
    forecast_payload: Optional[Dict[str, object]],
    hourly_payload: Optional[Dict[str, object]],
    observations_payload: Optional[Dict[str, object]],
    target_date: dt.date,
) -> ProviderSignal:
    mu_high = _forecast_mu_for_date(
        forecast_payload=forecast_payload or {},
        target_date=target_date,
        kind="HIGH",
        hourly_payload=hourly_payload,
        observed_payload=observations_payload,
    )
    mu_low = _forecast_mu_for_date(
        forecast_payload=forecast_payload or {},
        target_date=target_date,
        kind="LOW",
        hourly_payload=hourly_payload,
        observed_payload=observations_payload,
    )
    as_of_ts = _parse_forecast_timestamp(
        forecast_payload=forecast_payload,
        forecast_hourly_payload=hourly_payload,
        observations_payload=observations_payload,
    )
    return ProviderSignal(
        provider="nws",
        mu_high_f=mu_high,
        mu_low_f=mu_low,
        as_of_ts=as_of_ts,
        metadata={"source": "nws_api_hourly_obs"},
    )


def _fetch_accuweather_signal(
    *,
    lat: float,
    lon: float,
    target_date: dt.date,
    cache_dir: Path,
    ttl_minutes: float,
) -> Optional[ProviderSignal]:
    api_key = str(os.environ.get("ACCUWEATHER_API_KEY") or "").strip()
    if not api_key:
        return None
    geokey = f"{lat:.4f},{lon:.4f}"

    loc_cache = _provider_cache_path(cache_dir=cache_dir, provider="accuweather", cache_key=f"{geokey}_loc")
    daily_cache = _provider_cache_path(cache_dir=cache_dir, provider="accuweather", cache_key=f"{geokey}_daily")
    current_cache = _provider_cache_path(cache_dir=cache_dir, provider="accuweather", cache_key=f"{geokey}_current")

    location_payload = read_cached_json(loc_cache, ttl_minutes=ttl_minutes)
    daily_payload = read_cached_json(daily_cache, ttl_minutes=ttl_minutes)
    current_payload = read_cached_json(current_cache, ttl_minutes=ttl_minutes)

    try:
        with requests.Session() as s:
            if not isinstance(location_payload, dict):
                r = s.get(
                    "https://dataservice.accuweather.com/locations/v1/cities/geoposition/search",
                    params={"apikey": api_key, "q": f"{lat},{lon}"},
                    timeout=20.0,
                )
                r.raise_for_status()
                j = r.json()
                if not isinstance(j, dict):
                    return None
                location_payload = j
                write_cached_json(loc_cache, j)

            loc_key = str(location_payload.get("Key") or "").strip()
            if not loc_key:
                return None

            if not isinstance(daily_payload, dict):
                r = s.get(
                    f"https://dataservice.accuweather.com/forecasts/v1/daily/5day/{loc_key}",
                    params={"apikey": api_key, "metric": "false", "details": "true"},
                    timeout=20.0,
                )
                r.raise_for_status()
                j = r.json()
                if isinstance(j, dict):
                    daily_payload = j
                    write_cached_json(daily_cache, j)

            if not isinstance(current_payload, dict):
                r = s.get(
                    f"https://dataservice.accuweather.com/currentconditions/v1/{loc_key}",
                    params={"apikey": api_key, "details": "true"},
                    timeout=20.0,
                )
                r.raise_for_status()
                j = r.json()
                if isinstance(j, list):
                    current_payload = {"rows": j}
                    write_cached_json(current_cache, current_payload)
    except Exception:
        return None

    mu_high: Optional[float] = None
    mu_low: Optional[float] = None
    as_of_ts: Optional[dt.datetime] = None

    daily_rows = daily_payload.get("DailyForecasts") if isinstance(daily_payload, dict) else None
    if isinstance(daily_rows, list):
        for row in daily_rows:
            if not isinstance(row, dict):
                continue
            if _iso_date_part(row.get("Date")) != target_date.isoformat():
                continue
            temp = row.get("Temperature")
            if isinstance(temp, dict):
                tmax = temp.get("Maximum")
                tmin = temp.get("Minimum")
                if isinstance(tmax, dict):
                    mu_high = _safe_float(tmax.get("Value"))
                if isinstance(tmin, dict):
                    mu_low = _safe_float(tmin.get("Value"))
            t_as_of = _iso_to_datetime(str(row.get("EpochDate") or ""))
            if t_as_of is not None:
                as_of_ts = t_as_of
            break

    current_rows = current_payload.get("rows") if isinstance(current_payload, dict) else None
    current_temp: Optional[float] = None
    if isinstance(current_rows, list) and current_rows:
        cur = current_rows[0]
        if isinstance(cur, dict):
            temp_obj = cur.get("Temperature")
            if isinstance(temp_obj, dict):
                imp = temp_obj.get("Imperial")
                if isinstance(imp, dict):
                    current_temp = _safe_float(imp.get("Value"))
            cur_ts = _iso_to_datetime(str(cur.get("LocalObservationDateTime") or ""))
            if cur_ts is not None and (as_of_ts is None or cur_ts > as_of_ts):
                as_of_ts = cur_ts

    if target_date == dt.datetime.now(dt.timezone.utc).date() and current_temp is not None:
        if mu_high is None or current_temp > mu_high:
            mu_high = float(current_temp)
        if mu_low is None or current_temp < mu_low:
            mu_low = float(current_temp)

    if mu_high is None and mu_low is None:
        return None

    return ProviderSignal(
        provider="accuweather",
        mu_high_f=mu_high,
        mu_low_f=mu_low,
        as_of_ts=as_of_ts,
        metadata={"source": "accuweather_api"},
    )


def _fetch_twc_signal(
    *,
    lat: float,
    lon: float,
    target_date: dt.date,
    cache_dir: Path,
    ttl_minutes: float,
) -> Optional[ProviderSignal]:
    api_key = str(os.environ.get("TWC_API_KEY") or "").strip()
    if not api_key:
        return None

    base = str(os.environ.get("TWC_API_BASE") or "https://api.weather.com").strip().rstrip("/")
    geocode = f"{lat:.4f},{lon:.4f}"
    daily_cache = _provider_cache_path(cache_dir=cache_dir, provider="twc", cache_key=f"{geocode}_daily")
    current_cache = _provider_cache_path(cache_dir=cache_dir, provider="twc", cache_key=f"{geocode}_current")
    daily_payload = read_cached_json(daily_cache, ttl_minutes=ttl_minutes)
    current_payload = read_cached_json(current_cache, ttl_minutes=ttl_minutes)

    params = {
        "geocode": geocode,
        "format": "json",
        "units": "e",
        "language": "en-US",
        "apiKey": api_key,
    }
    try:
        with requests.Session() as s:
            if not isinstance(daily_payload, dict):
                r = s.get(f"{base}/v3/wx/forecast/daily/5day", params=params, timeout=20.0)
                r.raise_for_status()
                j = r.json()
                if isinstance(j, dict):
                    daily_payload = j
                    write_cached_json(daily_cache, j)
            if not isinstance(current_payload, dict):
                r = s.get(f"{base}/v3/wx/observations/current", params=params, timeout=20.0)
                r.raise_for_status()
                j = r.json()
                if isinstance(j, dict):
                    current_payload = j
                    write_cached_json(current_cache, j)
    except Exception:
        return None

    mu_high: Optional[float] = None
    mu_low: Optional[float] = None
    as_of_ts: Optional[dt.datetime] = None

    if isinstance(daily_payload, dict):
        valid = daily_payload.get("validTimeLocal")
        tmax = daily_payload.get("temperatureMax")
        tmin = daily_payload.get("temperatureMin")
        if isinstance(valid, list) and isinstance(tmax, list) and isinstance(tmin, list):
            n = min(len(valid), len(tmax), len(tmin))
            for i in range(n):
                if _iso_date_part(valid[i]) != target_date.isoformat():
                    continue
                mu_high = _safe_float(tmax[i])
                mu_low = _safe_float(tmin[i])
                ts = _iso_to_datetime(str(valid[i]))
                if ts is not None:
                    as_of_ts = ts
                break

    current_temp = _safe_float(current_payload.get("temperature")) if isinstance(current_payload, dict) else None
    if isinstance(current_payload, dict):
        ts = _safe_float(current_payload.get("validTimeUtc"))
        if ts is not None:
            try:
                cur_ts = dt.datetime.fromtimestamp(float(ts), tz=dt.timezone.utc)
                if as_of_ts is None or cur_ts > as_of_ts:
                    as_of_ts = cur_ts
            except Exception:
                pass

    if target_date == dt.datetime.now(dt.timezone.utc).date() and current_temp is not None:
        if mu_high is None or current_temp > mu_high:
            mu_high = float(current_temp)
        if mu_low is None or current_temp < mu_low:
            mu_low = float(current_temp)

    if mu_high is None and mu_low is None:
        return None

    return ProviderSignal(
        provider="twc",
        mu_high_f=mu_high,
        mu_low_f=mu_low,
        as_of_ts=as_of_ts,
        metadata={"source": "twc_api"},
    )


class WeatherSpecialist:
    name = "weather"
    categories = ["Climate and Weather"]

    def _cache_dir(self, context: RunContext) -> Path:
        return context.data_root / "external" / "weather"

    def propose(self, context: RunContext) -> List[CandidateProposal]:
        out: List[CandidateProposal] = []
        cfg = context.config
        join_ticks = int(cfg.get("weather_join_ticks", 1))
        slippage_cents = float(cfg.get("weather_slippage_cents_per_contract", 0.25))
        min_ev = float(cfg.get("weather_min_ev_dollars", 0.0))
        cache_ttl = float(cfg.get("weather_cache_ttl_minutes", 30.0))
        calibration_ttl = float(cfg.get("weather_calibration_ttl_minutes", max(cache_ttl, 60.0)))
        user_agent = str(
            cfg.get("weather_nws_user_agent", "Chimera_v4_NightWatch (kellypclarke-maker@github.com)")
        ).strip()
        maker_fee_rate = float(cfg.get("maker_fee_rate", 0.0))
        min_yes_bid_cents = int(cfg.get("min_yes_bid_cents", 1))
        max_yes_spread_cents = int(cfg.get("max_yes_spread_cents", 10))
        weather_max_spread_cents = int(cfg.get("weather_max_spread_cents", max_yes_spread_cents))
        weather_min_model_edge = float(cfg.get("weather_min_model_edge_per_contract", 0.0))
        trade_high_enabled = bool(cfg.get("weather_trade_high_enabled", True))
        trade_low_enabled = bool(cfg.get("weather_trade_low_enabled", True))
        use_det_locks = bool(cfg.get("weather_use_deterministic_extrema_locks", True))
        ensemble_enabled = bool(cfg.get("weather_enable_multi_source", False))
        provider_weights = _parse_weight_map(cfg.get("weather_provider_weights"))
        if not provider_weights:
            provider_weights = {"nws": 1.0}
        max_ensemble_disagreement = max(0.0, float(cfg.get("weather_ensemble_max_disagreement_f", 0.0)))
        sigma_disagreement_mult = max(0.0, float(cfg.get("weather_ensemble_sigma_disagreement_mult_per_f", 0.0)))
        cache_dir = self._cache_dir(context)
        cache_dir.mkdir(parents=True, exist_ok=True)
        calibration_profile = _load_weather_calibration_profile(
            config=cfg,
            cache_dir=cache_dir,
            ttl_minutes=calibration_ttl,
        )

        for row in context.markets:
            if category_from_row(row) != "Climate and Weather":
                continue
            ticker = str(row.get("ticker") or "").strip().upper()
            m = WEATHER_MARKET_RE.match(ticker)
            if not m:
                continue

            kind = str(m.group("kind") or "").upper()
            code = str(m.group("code") or "").upper()
            date_token = str(m.group("date") or "").upper()
            target_date = _parse_date(date_token)
            if target_date is None:
                continue
            if kind == "HIGH" and not trade_high_enabled:
                continue
            if kind == "LOW" and not trade_low_enabled:
                continue

            bounds = _parse_bucket_bounds(str(row.get("yes_sub_title") or ""))
            if bounds is None:
                continue
            lower, upper = bounds
            bucket_kind = _bucket_kind(lower=lower, upper=upper)
            cmp_text = " ".join(
                [
                    str(row.get("rules_primary") or ""),
                    str(row.get("title") or ""),
                ]
            )
            cmp_hints = _infer_comparator_hints(cmp_text)
            if not _bucket_matches_comparator(bucket_kind=bucket_kind, hints=cmp_hints):
                rules_pointer, rules_missing = rules_pointer_from_row(row)
                out.append(
                    CandidateProposal(
                        strategy_id="WTHR_temp_bucket_diagnostic",
                        ticker=ticker,
                        title=str(row.get("title") or ""),
                        category=category_from_row(row),
                        close_time=str(row.get("close_time") or ""),
                        event_ticker=str(row.get("event_ticker") or ""),
                        side="yes",
                        action="diagnostic_only",
                        maker_or_taker="maker",
                        yes_bid=safe_int(row.get("yes_bid")),
                        yes_ask=safe_int(row.get("yes_ask")),
                        no_bid=safe_int(row.get("no_bid")),
                        no_ask=safe_int(row.get("no_ask")),
                        per_contract_notional=0.0,
                        size_contracts=0,
                        liquidity_notes=str(row.get("yes_sub_title") or ""),
                        risk_flags=f"bucket_comparator_mismatch:bucket={bucket_kind};hints={','.join(sorted(cmp_hints))}",
                        verification_checklist="verify rules/title comparator and bucket text alignment before trading",
                        rules_text_hash=rules_hash_from_row(row),
                        rules_missing=rules_missing,
                        rules_pointer=rules_pointer,
                        resolution_pointer=resolution_pointer_from_row(row),
                        market_url=str(row.get("market_url") or ""),
                    )
                )
                continue

            rules_primary = str(row.get("rules_primary") or "")
            rules_secondary = str(row.get("rules_secondary") or "")
            station_ctx = _resolve_station_context(
                rules_primary=rules_primary,
                rules_secondary=rules_secondary,
                title=str(row.get("title") or ""),
                ticker_code=code,
                cache_dir=cache_dir,
                ttl_minutes=cache_ttl,
                user_agent=user_agent,
            )
            if station_ctx is None:
                continue

            station_id = str(station_ctx.station_id).strip().upper()
            station_name = str(station_ctx.station_name)
            station_coords = (float(station_ctx.station_lat), float(station_ctx.station_lon))

            forecast = _fetch_nws_forecast(
                code=station_id,
                lat=float(station_coords[0]),
                lon=float(station_coords[1]),
                cache_dir=cache_dir,
                ttl_minutes=cache_ttl,
                user_agent=user_agent,
            )
            forecast_hourly = _fetch_nws_hourly_forecast(
                code=station_id,
                lat=float(station_coords[0]),
                lon=float(station_coords[1]),
                cache_dir=cache_dir,
                ttl_minutes=cache_ttl,
                user_agent=user_agent,
            )
            observations = _fetch_station_observations(
                station_id=station_id,
                cache_dir=cache_dir,
                ttl_minutes=cache_ttl,
                user_agent=user_agent,
            )

            nws_signal = _build_nws_signal(
                forecast_payload=forecast,
                hourly_payload=forecast_hourly,
                observations_payload=observations,
                target_date=target_date,
            )
            mu_nws = _provider_mu_for_kind(signal=nws_signal, kind=kind)
            if mu_nws is None:
                continue
            mu = float(mu_nws)
            source = "nws_api_hourly_obs"
            providers_used = ["nws"]
            provider_weights_used: Dict[str, float] = {"nws": 1.0}
            ensemble_disagreement_f = 0.0
            ensemble_as_of_ts = nws_signal.as_of_ts

            provider_signals: List[ProviderSignal] = [nws_signal]
            if ensemble_enabled:
                aw_signal = _fetch_accuweather_signal(
                    lat=float(station_coords[0]),
                    lon=float(station_coords[1]),
                    target_date=target_date,
                    cache_dir=cache_dir,
                    ttl_minutes=cache_ttl,
                )
                if aw_signal is not None:
                    provider_signals.append(aw_signal)
                twc_signal = _fetch_twc_signal(
                    lat=float(station_coords[0]),
                    lon=float(station_coords[1]),
                    target_date=target_date,
                    cache_dir=cache_dir,
                    ttl_minutes=cache_ttl,
                )
                if twc_signal is not None:
                    provider_signals.append(twc_signal)

                if len(provider_signals) > 1:
                    mu_ensemble, ensemble_meta = _weighted_ensemble_mu(
                        signals=provider_signals,
                        kind=kind,
                        weights=provider_weights,
                    )
                    if mu_ensemble is not None:
                        disagreement = _safe_float(ensemble_meta.get("disagreement_f")) or 0.0
                        if max_ensemble_disagreement > 0.0 and disagreement > max_ensemble_disagreement:
                            continue
                        mu = float(mu_ensemble)
                        source = "ensemble"
                        providers_used = [str(x) for x in ensemble_meta.get("providers") or []]
                        if not providers_used:
                            providers_used = ["nws"]
                        weights_raw = ensemble_meta.get("weights")
                        provider_weights_used = (
                            {str(k): float(v) for k, v in weights_raw.items()} if isinstance(weights_raw, dict) else {"nws": 1.0}
                        )
                        ensemble_disagreement_f = float(disagreement)
                        ens_ts = _iso_to_datetime(str(ensemble_meta.get("as_of_ts") or ""))
                        if ens_ts is not None:
                            ensemble_as_of_ts = ens_ts

            forecast_as_of_ts = ensemble_as_of_ts or _parse_forecast_timestamp(
                forecast_payload=forecast,
                forecast_hourly_payload=forecast_hourly,
                observations_payload=observations,
            )
            local_tz = _resolve_local_tz(
                target_date=target_date,
                hourly_payload=forecast_hourly,
                forecast_payload=forecast or {},
            )
            observed_temps_today = _observation_temps_for_date(
                observed_payload=observations,
                target_date=target_date,
                local_tz=local_tz,
            )
            sigma, sigma_meta = _effective_weather_sigma(
                kind=kind,
                code=code,
                station_id=station_id,
                target_date=target_date,
                local_tz=local_tz,
                as_of_ts=forecast_as_of_ts,
                config=cfg,
                cache_dir=cache_dir,
                cache_ttl_minutes=cache_ttl,
            )
            if ensemble_disagreement_f > 0.0 and sigma_disagreement_mult > 0.0:
                sigma_floor = max(0.25, float(cfg.get("weather_sigma_floor_f", 1.5)))
                sigma_cap = max(sigma_floor, float(cfg.get("weather_sigma_cap_f", 8.0)))
                sigma_mult = 1.0 + (sigma_disagreement_mult * float(ensemble_disagreement_f))
                sigma = min(sigma_cap, max(sigma_floor, float(sigma) * float(sigma_mult)))
                sigma_meta["ensemble_sigma_mult"] = float(sigma_mult)
            else:
                sigma_meta["ensemble_sigma_mult"] = 1.0
            sigma_meta["ensemble_disagreement_f"] = float(ensemble_disagreement_f)

            p_true_raw = _bucket_probability(lower=lower, upper=upper, mu=float(mu), sigma=float(sigma))
            if not (0.0 <= p_true_raw <= 1.0):
                continue
            det_lock_applied = ""
            p_true_locked = float(p_true_raw)
            if use_det_locks:
                det_lock = _deterministic_bucket_probability_from_extrema(
                    kind=kind,
                    lower=lower,
                    upper=upper,
                    observed_temps_f=observed_temps_today,
                )
                if det_lock is not None:
                    p_true_locked = float(det_lock)
                    det_lock_applied = "yes"
            lead_hours = float(sigma_meta.get("lead_hours") or 0.0)
            if det_lock_applied:
                p_true = _clamp01(p_true_locked)
                cal_meta = {"segment": "det_lock", "method": "deterministic"}
            else:
                p_true, cal_meta = _calibrate_weather_probability(
                    p_true_raw=float(p_true_locked),
                    kind=kind,
                    station_id=station_id,
                    lead_hours=lead_hours,
                    profile=calibration_profile,
                )
            if not (0.0 <= p_true <= 1.0):
                continue

            yes_bid = safe_int(row.get("yes_bid"))
            yes_ask = safe_int(row.get("yes_ask"))
            if not passes_maker_liquidity(
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                min_yes_bid_cents=min_yes_bid_cents,
                max_yes_spread_cents=max_yes_spread_cents,
            ):
                continue
            spread_cents = max(0, int(yes_ask or 0) - max(0, int(yes_bid or 0)))
            if spread_cents > max(0, int(weather_max_spread_cents)):
                continue
            intended = intended_maker_yes_price(yes_bid=yes_bid, yes_ask=yes_ask, join_ticks=join_ticks)
            if intended is None:
                continue

            p_fill = maker_fill_prob(yes_bid, yes_ask, intended)
            if p_fill <= 0.0:
                continue
            price = float(intended) / 100.0
            model_edge = float(p_true) - float(price)
            if model_edge < float(weather_min_model_edge):
                continue
            fees = maker_fee_dollars(contracts=1, price=price, rate=maker_fee_rate)
            slippage = float(slippage_cents) / 100.0
            ev = float(p_fill) * (float(p_true) - price - fees - slippage)
            if ev <= float(min_ev):
                continue

            rules_pointer, rules_missing = rules_pointer_from_row(row)
            forecast_ts = ensemble_as_of_ts.isoformat() if ensemble_as_of_ts is not None else ""
            if not forecast_ts:
                forecast_props = forecast.get("properties") if isinstance(forecast, dict) else {}
                hourly_props = forecast_hourly.get("properties") if isinstance(forecast_hourly, dict) else {}
                if isinstance(forecast_props, dict):
                    forecast_ts = str(forecast_props.get("updated") or forecast_props.get("generatedAt") or "").strip()
                if not forecast_ts and isinstance(hourly_props, dict):
                    forecast_ts = str(hourly_props.get("updated") or hourly_props.get("generatedAt") or "").strip()
                if not forecast_ts:
                    forecast_ts = _latest_observation_timestamp(observations)

            providers_token = ",".join(str(p) for p in providers_used if str(p).strip())
            weights_token = ",".join(
                f"{str(k)}:{float(v):.4f}" for k, v in sorted(provider_weights_used.items(), key=lambda kv: str(kv[0]))
            )

            out.append(
                CandidateProposal(
                    strategy_id="WTHR_nws_temp_bucket_maker",
                    ticker=ticker,
                    title=str(row.get("title") or ""),
                    category=category_from_row(row),
                    close_time=str(row.get("close_time") or ""),
                    event_ticker=str(row.get("event_ticker") or ""),
                    side="yes",
                    action="post_yes",
                    maker_or_taker="maker",
                    yes_price_cents=intended,
                    no_price_cents=None,
                    yes_bid=yes_bid,
                    yes_ask=yes_ask,
                    no_bid=safe_int(row.get("no_bid")),
                    no_ask=safe_int(row.get("no_ask")),
                    sum_prices_cents=None,
                    p_fill_assumed=float(p_fill),
                    fees_assumed_dollars=float(fees),
                    slippage_assumed_dollars=float(slippage),
                    ev_dollars=float(ev),
                    ev_pct=(float(ev) / price if price > 0 else float("nan")),
                    per_contract_notional=1.0,
                    size_contracts=1,
                    liquidity_notes=(
                        f"p_true={p_true:.4f};p_true_raw={p_true_raw:.4f};p_true_cal={p_true:.4f};"
                        f"mu={mu:.2f};sigma={sigma:.2f};source={source};"
                        f"forecast_ts={forecast_ts};target_date={target_date.isoformat()};"
                        f"providers={providers_token};weights={weights_token};"
                        f"ensemble_disagreement_f={float(ensemble_disagreement_f):.4f};"
                        f"station_id={station_id};station_name={station_name};"
                        f"station_lat={station_coords[0]:.4f};station_lon={station_coords[1]:.4f};"
                        f"station_source={station_ctx.source};cli_code={station_ctx.cli_code};"
                        f"sigma_lead_h={float(sigma_meta.get('lead_hours') or 0.0):.2f};"
                        f"sigma_season={str(sigma_meta.get('season') or '')};"
                        f"sigma_lead_mult={float(sigma_meta.get('lead_mult') or 1.0):.4f};"
                        f"sigma_station_mult={float(sigma_meta.get('station_mult') or 1.0):.4f};"
                        f"sigma_profile_station_mult={float(sigma_meta.get('profile_station_mult') or 1.0):.4f};"
                        f"sigma_profile_season_mult={float(sigma_meta.get('profile_season_mult') or 1.0):.4f};"
                        f"sigma_ensemble_mult={float(sigma_meta.get('ensemble_sigma_mult') or 1.0):.4f};"
                        f"cal_segment={str(cal_meta.get('segment') or '')};"
                        f"cal_method={str(cal_meta.get('method') or '')};"
                        f"det_lock={det_lock_applied}"
                    ),
                    risk_flags=(
                        f"model=normal_temp_dynamic_sigma_calibrated;code={code};kind={kind};station_id={station_id};"
                        f"station_source={station_ctx.source};cli_code={station_ctx.cli_code};source={source}"
                    ),
                    verification_checklist=(
                        "verify ticker date/location parse; verify provider freshness and disagreement guardrail; "
                        "verify bucket text parsing; verify calibration profile age; verify maker fill/fee/slippage assumptions"
                    ),
                    rules_text_hash=rules_hash_from_row(row),
                    rules_missing=rules_missing,
                    rules_pointer=rules_pointer,
                    resolution_pointer=resolution_pointer_from_row(row),
                    market_url=str(row.get("market_url") or ""),
                )
            )
        return out
