"""Feature engineering for the routine-based fatigue recommender.

Design notes
------------
* Features describe the crew member's *routine and context* (schedule disruption, duty length,
  sleep/rest problems, circadian shift, time of day, flight phase, role, aircraft family ...).
  They are extracted from the structured ASRS fields and from keyword patterns in the narrative.
* The target ("risk_level") is a proxy for the *operational consequence* of the fatigue-related
  event, derived ONLY from the Anomaly/Result fields. Those fields are never used as inputs, so
  the models cannot simply read the label off the features.
* The same binary routine flags can be produced from a questionnaire (see recommend.py), so
  the trained models can be applied to a new user's routine.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- routine flags
FLAG_PATTERNS: dict[str, str] = {
    "sched_disruption": r"revision|revised|reassign|swap|re-?rout|reserve|crew swap|scheduling (told|changed)",
    "long_duty": (
        r"long day|long duty|duty period|duty time|\bfdp\b|timed out|extension|extend|"
        r"\b\d{2} hours? of duty|1[0-9][ -]hour|four legs|4 legs|\bleg [45]\b|day 4|last leg|"
        r"14 hours|12 hours|8\.5 hours"
    ),
    "delays": r"delay|hold times?|waiting|waited|irops|cancel",
    "sleep_rest_issue": (
        r"sleep|slept|rest period|crew rest|hotel|did not get into bed|poor night|not eaten|"
        r"didn.t eat|hadn.t eaten|exhaust"
    ),
    "circadian_shift": (
        r"night schedule|day schedule|body clock|late nights? to early|early mornings?|red-?eye|"
        r"circadian|started very early|swap\w* .{0,25}schedule"
    ),
    "commute_travel": r"commut|deadhead|rideshare|jumpseat|\bvan\b",
    "weather_disruption": r"thunderstorm|storm|snow|icing|de-?ic|severe weather|turbulence",
    "workload_pressure": r"rushed|task saturat|workload|time pressure|get-?there-?itis|distract|busy|pressure",
    "unfamiliarity": r"new to|first (time|flight)|unfamiliar|new airport|inexperience|never seen",
}
FLAG_COLS = list(FLAG_PATTERNS)

# --------------------------------------------------------------------------- context columns
CATEGORICAL_COLS = ["time_bucket", "light_cond", "phase_group", "role", "aircraft_family"]
NUMERIC_COLS = ["crew_size_num"]
FEATURE_COLS = FLAG_COLS + CATEGORICAL_COLS + NUMERIC_COLS

TIME_BUCKETS = ["night", "morning", "afternoon", "evening", "unknown"]
LIGHT_CONDS = ["Daylight", "Dawn", "Dusk", "Night", "unknown"]
PHASE_GROUPS = ["ground", "departure_climb", "cruise", "descent_approach", "unknown"]
ROLES = ["captain", "first_officer", "other"]
AIRCRAFT_FAMILIES = ["boeing", "airbus", "regional_jet", "other"]

RISK_LEVELS = ["Low", "Medium", "High"]


def _time_bucket(s: str) -> str:
    return {
        "0001-0600": "night",
        "0601-1200": "morning",
        "1201-1800": "afternoon",
        "1801-2400": "evening",
    }.get(str(s).strip(), "unknown")


def _phase_group(s: str) -> str:
    s = str(s).lower()
    if not s or s == "nan":
        return "unknown"
    if any(k in s for k in ("parked", "taxi")):
        return "ground"
    if any(k in s for k in ("takeoff", "climb", "initial climb")):
        return "departure_climb"
    if "cruise" in s:
        return "cruise"
    if any(k in s for k in ("descent", "approach", "landing")):
        return "descent_approach"
    return "unknown"


def _role(s: str) -> str:
    s = str(s).lower()
    if "captain" in s:
        return "captain"
    if "first officer" in s or "relief pilot" in s:
        return "first_officer"
    return "other"


def _aircraft_family(s: str) -> str:
    s = str(s).upper()
    if re.search(r"\bB7\d\d|BOEING|\bB737|MD-?\d", s):
        return "boeing"
    if re.search(r"\bA3\d\d|AIRBUS|\bA\d{3}", s):
        return "airbus"
    if re.search(r"EMB|CRJ|REGIONAL|ERJ", s):
        return "regional_jet"
    return "other"


def _light(s: str) -> str:
    s = str(s).strip()
    return s if s in LIGHT_CONDS else "unknown"


def extract_flags(text: str) -> dict[str, int]:
    t = str(text).lower()
    return {name: int(bool(re.search(pat, t))) for name, pat in FLAG_PATTERNS.items()}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the parsed CSV (see parse_asrs.py) into the model feature table."""
    out = pd.DataFrame(index=df.index)
    flags = df["narrative"].fillna("").apply(extract_flags).apply(pd.Series)
    for c in FLAG_COLS:
        out[c] = flags[c].astype(int)

    out["time_bucket"] = df["local_time"].apply(_time_bucket)
    out["light_cond"] = df["light"].apply(_light)
    out["phase_group"] = df["flight_phase"].apply(_phase_group)
    out["role"] = df["functions"].apply(_role)
    out["aircraft_family"] = df["aircraft"].apply(_aircraft_family)
    out["crew_size_num"] = pd.to_numeric(df["crew_size"], errors="coerce").fillna(2).astype(float)
    return out[FEATURE_COLS]


# --------------------------------------------------------------------------- target label
_HIGH = (
    r"diverted|landed in emergency|returned to departure|regained aircraft control|"
    r"executed go around|took evasive action|critical|cftt|loss of aircraft control|"
    r"conflict|rejected takeoff|ground incursion"
)
_MEDIUM = (
    r"unstabilized|deviation - |clearance|ground event|less severe|"
    r"flight cancelled|returned to gate|overrode automation"
)


def derive_risk_level(row: pd.Series) -> str:
    """Proxy label: operational consequence severity of the report (Low / Medium / High)."""
    text = " ".join(str(row.get(c, "")) for c in ("anomalies", "anomaly_detail", "results")).lower()
    if re.search(_HIGH, text):
        return "High"
    if re.search(_MEDIUM, text):
        return "Medium"
    return "Low"


def add_labels(df: pd.DataFrame) -> pd.Series:
    return df.apply(derive_risk_level, axis=1)


def feature_names_for_report(columns: list[str]) -> list[str]:
    """Strip the sklearn ColumnTransformer prefixes for nicer feature-importance output."""
    return [re.sub(r"^(cat|num)__", "", c) for c in columns]


def describe_flags(row: pd.Series) -> list[str]:
    return [c for c in FLAG_COLS if int(row.get(c, 0)) == 1]


if __name__ == "__main__":  # quick manual check
    import sys

    df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/processed/asrs_fatigue.csv").fillna("")
    X = build_features(df)
    y = add_labels(df)
    print(X[FLAG_COLS].mean().round(2).to_string())
    print(y.value_counts())
    np.set_printoptions(linewidth=120)
