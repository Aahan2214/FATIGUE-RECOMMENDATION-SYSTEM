"""Routine-based fatigue & health recommendation engine.

How it works
------------
1. A crew member's routine (questionnaire) is converted to the same feature vector used in training.
2. Random Forest  -> estimated risk level (Low / Medium / High) with class probabilities.
3. KNN            -> the k most similar historical ASRS fatigue reports (case-based reasoning).
4. Recommendations are scored by combining
       (a) routine factors present in the user's own profile      (weight 1.0)
       (b) routine factors frequent among the similar reports      (weight 0.5, similarity-weighted)
   and mapped to general fatigue-management countermeasures.


"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import FEATURE_COLS, FLAG_COLS, RISK_LEVELS

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

DISCLAIMER = (
    "Educational decision-support built from a small set of voluntary ASRS reports. "
    "Not medical or regulatory advice - follow 14 CFR Part 117 and your operator's fatigue "
    "risk management procedures."
)

# --------------------------------------------------------------------------- knowledge base
COUNTERMEASURES: dict[str, dict] = {
    "sleep_rest_issue": {
        "title": "Protect sleep and nutrition",
        "advice": [
            "Make the next sleep opportunity the priority: dark, cool, quiet room; phone silenced.",
            "If operator procedures allow it, use planned/controlled rest or a short nap (about 30 min "
            "or less) to limit sleep inertia, and allow time to wake fully before taking over duties.",
            "Eat regular balanced meals and keep hydrated; avoid caffeine within ~6 h of your sleep period.",
        ],
    },
    "long_duty": {
        "title": "Re-assess fitness for duty during long duty days",
        "advice": [
            "Self-assess before every leg and before accepting any duty extension - ask how you will "
            "feel 2-3 hours from now, not only right now.",
            "Plan breaks, light snacks, and movement between legs; save demanding tasks for when you are sharpest.",
            "Be prepared to call fatigued through your company's fatigue-reporting process.",
        ],
    },
    "circadian_shift": {
        "title": "Manage circadian disruption",
        "advice": [
            "Avoid abrupt day/night swaps where possible; shift sleep timing gradually before a pairing.",
            "Use bright light when you need to be alert and dark/sunglasses when you need to sleep.",
            "Expect the lowest alertness roughly between 02:00-06:00 body-clock time - brief extra, "
            "and cross-check critical tasks, during that window.",
        ],
    },
    "sched_disruption": {
        "title": "Handle schedule changes and revisions",
        "advice": [
            "After a trip revision, re-confirm your legal rest and plan sleep before reporting.",
            "Tell scheduling early if a change leaves too little rest; document it.",
            "Avoid stacking minimum-rest periods back-to-back when you have a choice.",
        ],
    },
    "delays": {
        "title": "Cope with delays and irregular operations",
        "advice": [
            "Use long delays to rest, eat and hydrate rather than staying 'on' the whole time.",
            "Reset the crew's plan after each disruption: re-brief threats before the next phase.",
            "Reassess duty limits early so decisions about extensions are not made when exhausted.",
        ],
    },
    "commute_travel": {
        "title": "Reduce commuting and travel fatigue",
        "advice": [
            "Commute the day before or arrange rest/crash-pad access instead of commuting the same morning.",
            "Do not count travel time as rest; budget it into your sleep plan.",
        ],
    },
    "weather_disruption": {
        "title": "Weather and threat load",
        "advice": [
            "Brief weather and alternates earlier; take conservative options (extra fuel, earlier diversion).",
            "Avoid compressing decision time when tired - ask ATC for more time or a hold if needed.",
        ],
    },
    "workload_pressure": {
        "title": "Workload and time pressure",
        "advice": [
            "Slow down and trade speed for accuracy; say 'I'm task-saturated' and re-distribute tasks.",
            "Use standard callouts, checklists and the go-around option; fatigued crews are more likely to continue unstable approaches.",
        ],
    },
    "unfamiliarity": {
        "title": "Unfamiliar airport or aircraft while fatigued",
        "advice": [
            "Allow extra briefing time; review company notes and chart cautions in detail and discuss them aloud.",
            "Invite challenge: ask the other pilot to cross-check altitudes, modes and clearances.",
        ],
    },
}

RISK_ADVICE = {
    "High": "Your routine resembles reports that ended in serious events. Consider whether you are fit "
            "for duty and use your company's fatigue call-out process before operating.",
    "Medium": "Your routine resembles reports with deviations or procedural errors. Add buffers, "
              "cross-checks and rest before critical phases of flight.",
    "Low": "Your routine resembles lower-consequence reports, but fatigue still degrades performance - "
           "keep monitoring how you feel.",
}


# --------------------------------------------------------------------------- questionnaire -> features
def profile_from_routine(
    sleep_hours_24h: float,
    hours_on_duty: float,
    legs_flown: int,
    time_bucket: str,
    light_cond: str,
    phase_group: str,
    role: str,
    aircraft_family: str,
    crew_size: int = 2,
    schedule_changed: bool = False,
    flipped_day_night: bool = False,
    significant_delay: bool = False,
    commuted_or_deadheaded: bool = False,
    weather_disruption: bool = False,
    high_workload: bool = False,
    unfamiliar_airport_or_aircraft: bool = False,
) -> dict:
    """Map a routine questionnaire to the model's feature space."""
    return {
        "sched_disruption": int(schedule_changed),
        "long_duty": int(hours_on_duty >= 10 or legs_flown >= 4),
        "delays": int(significant_delay),
        "sleep_rest_issue": int(sleep_hours_24h < 6),
        "circadian_shift": int(flipped_day_night or time_bucket == "night"),
        "commute_travel": int(commuted_or_deadheaded),
        "weather_disruption": int(weather_disruption),
        "workload_pressure": int(high_workload),
        "unfamiliarity": int(unfamiliar_airport_or_aircraft),
        "time_bucket": time_bucket,
        "light_cond": light_cond,
        "phase_group": phase_group,
        "role": role,
        "aircraft_family": aircraft_family,
        "crew_size_num": float(crew_size),
    }


# --------------------------------------------------------------------------- engine
@dataclass
class Assessment:
    risk_rf: str
    rf_probabilities: dict[str, float]
    risk_knn: str
    neighbours: pd.DataFrame
    recommendations: list[dict] = field(default_factory=list)
    disclaimer: str = DISCLAIMER


class FatigueRecommender:
    def __init__(self, model_dir: Path = MODEL_DIR, k: int = 5):
        self.rf = joblib.load(model_dir / "rf.joblib")
        self.knn = joblib.load(model_dir / "knn.joblib")
        self.reference: pd.DataFrame = joblib.load(model_dir / "reference.joblib")
        self.k = min(k, len(self.reference))

    def _neighbours(self, X: pd.DataFrame) -> pd.DataFrame:
        prep = self.knn.named_steps["prep"]
        knn = self.knn.named_steps["knn"]
        dist, idx = knn.kneighbors(prep.transform(X), n_neighbors=self.k)
        nb = self.reference.iloc[idx[0]].copy()
        nb["distance"] = dist[0]
        nb["similarity"] = 1.0 / (1.0 + dist[0])
        return nb

    def assess(self, profile: dict, top_n: int = 5) -> Assessment:
        X = pd.DataFrame([profile])[FEATURE_COLS]

        proba = self.rf.predict_proba(X)[0]
        classes = list(self.rf.classes_)
        rf_probs = {c: float(p) for c, p in zip(classes, proba)}
        risk_rf = classes[int(np.argmax(proba))]
        risk_knn = str(self.knn.predict(X)[0])

        nb = self._neighbours(X)
        w = nb["similarity"].to_numpy()
        w = w / w.sum() if w.sum() else np.ones(len(w)) / len(w)

        scores: dict[str, float] = {}
        for flag in FLAG_COLS:
            own = float(profile.get(flag, 0))
            neighbour_share = float((nb[flag].to_numpy() * w).sum())
            scores[flag] = 1.0 * own + 0.5 * neighbour_share

        recs = []
        for flag, sc in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
            if sc < 0.25 or len(recs) >= top_n:
                continue
            cm = COUNTERMEASURES[flag]
            recs.append(
                {
                    "factor": flag,
                    "title": cm["title"],
                    "advice": cm["advice"],
                    "score": round(sc, 2),
                    "source": "your routine" if profile.get(flag, 0) else "similar past reports",
                }
            )
        recs.insert(0, {"factor": "overall", "title": f"Overall risk: {risk_rf}",
                        "advice": [RISK_ADVICE[risk_rf]], "score": 1.0, "source": "Random Forest"})
        out_cols = [c for c in ("acn", "risk_level", "similarity", "synopsis") if c in nb]
        return Assessment(risk_rf, rf_probs, risk_knn, nb[out_cols].reset_index(drop=True), recs)


if __name__ == "__main__":
    rec = FatigueRecommender()
    demo = profile_from_routine(
        sleep_hours_24h=4.5, hours_on_duty=11, legs_flown=4, time_bucket="evening",
        light_cond="Night", phase_group="descent_approach", role="captain",
        aircraft_family="regional_jet", schedule_changed=True, significant_delay=True,
    )
    a = rec.assess(demo)
    print("RF risk:", a.risk_rf, a.rf_probabilities, "| KNN:", a.risk_knn)
    for r in a.recommendations:
        print(f"- [{r['source']}] {r['title']}")
