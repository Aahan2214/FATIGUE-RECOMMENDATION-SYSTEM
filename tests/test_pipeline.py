"""Run with:  pytest -q   (or  python tests/test_pipeline.py )"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import FEATURE_COLS, FLAG_COLS, add_labels, build_features  # noqa: E402
from src.parse_asrs import parse_report_set  # noqa: E402
from src.recommend import FatigueRecommender, profile_from_routine  # noqa: E402
from src.train import run  # noqa: E402

SAMPLE = (ROOT / "tests" / "sample_asrs.txt").read_text()


def test_parser_extracts_records_and_fields():
    df = parse_report_set(SAMPLE)
    assert len(df) == 3                                  # synopsis list is ignored
    assert df.loc[0, "acn"] == "1000001"
    assert df.loc[0, "flight_phase"] == "Initial Approach"
    assert "Fatigue" in df.loc[0, "human_factors"]
    assert "Captain" in df.loc[0, "functions"]
    assert "Took Evasive Action" in df.loc[0, "results"]
    assert "Second crew member" in df.loc[1, "narrative"]  # multi-narrative merged
    assert df.loc[2, "synopsis"] == "Minor ground event."


def test_features_and_labels():
    df = parse_report_set(SAMPLE).fillna("")
    X = build_features(df)
    assert list(X.columns) == FEATURE_COLS
    assert X.loc[0, "sched_disruption"] == 1 and X.loc[0, "delays"] == 1 and X.loc[0, "long_duty"] == 1
    assert X.loc[1, "sleep_rest_issue"] == 1
    assert X.loc[2, "circadian_shift"] == 1 and X.loc[2, "weather_disruption"] == 1
    assert X.loc[2, "phase_group"] == "ground" and X.loc[1, "aircraft_family"] == "regional_jet"
    assert list(add_labels(df)) == ["High", "Medium", "Medium"]


def _synthetic(n=90, seed=0) -> pd.DataFrame:
    """Random data purely to smoke-test the code path (NOT used for any reported result)."""
    rng = np.random.default_rng(seed)
    base = parse_report_set(SAMPLE).fillna("")
    rows = []
    for i in range(n):
        r = base.iloc[i % 3].copy()
        r["acn"] = str(2000000 + i)
        r["local_time"] = rng.choice(["0001-0600", "0601-1200", "1201-1800", "1801-2400"])
        r["flight_phase"] = rng.choice(["Initial Approach", "Cruise", "Taxi", "Climb"])
        r["narrative"] += " " + " ".join(rng.choice(
            ["delay", "slept poorly", "storm", "rushed", "new to airport", "long day", "deadhead"],
            size=rng.integers(0, 4)))
        r["results"] = rng.choice(["Diverted", "", "Returned To Gate"])
        r["anomalies"] = rng.choice(["Unstabilized Approach", "", "Deviation - Altitude"])
        rows.append(r)
    return pd.DataFrame(rows).reset_index(drop=True)


def test_training_and_recommendation_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        metrics = run(_synthetic(), model_dir=tmp, repeats=1, verbose=False)
        assert set(metrics["cv"]) == {"Majority baseline", "KNN", "Random Forest"}
        rec = FatigueRecommender(model_dir=tmp)
        prof = profile_from_routine(
            sleep_hours_24h=4, hours_on_duty=12, legs_flown=4, time_bucket="night",
            light_cond="Night", phase_group="descent_approach", role="captain",
            aircraft_family="boeing", schedule_changed=True, significant_delay=True)
        a = rec.assess(prof)
        assert a.risk_rf in ("Low", "Medium", "High")
        assert abs(sum(a.rf_probabilities.values()) - 1) < 1e-6
        assert len(a.neighbours) == 5
        factors = {r["factor"] for r in a.recommendations}
        assert {"sleep_rest_issue", "long_duty", "sched_disruption"} <= factors


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
