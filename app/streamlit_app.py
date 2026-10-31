"""Streamlit front-end:  streamlit run app/streamlit_app.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.features import AIRCRAFT_FAMILIES, FLAG_COLS, LIGHT_CONDS, PHASE_GROUPS, ROLES, TIME_BUCKETS  # noqa: E402
from src.recommend import DISCLAIMER, FatigueRecommender, profile_from_routine  # noqa: E402

MODEL_DIR = ROOT / "models"
CSV = ROOT / "data" / "processed" / "asrs_fatigue.csv"

st.set_page_config(page_title="Aviation Fatigue Recommender", page_icon="✈️", layout="wide")


@st.cache_resource(show_spinner="Preparing models (first run only)...")
def load_recommender() -> FatigueRecommender:
    if not (MODEL_DIR / "rf.joblib").exists():
        if not CSV.exists():
            st.error(
                "No trained models and no data/processed/asrs_fatigue.csv found. Run "
                "`python scripts/download_data.py && python -m src.parse_asrs && python -m src.train`."
            )
            st.stop()
        from src.train import run

        run(pd.read_csv(CSV), repeats=2, verbose=False)
    return FatigueRecommender()


rec = load_recommender()

st.title("✈️ Routine-Based Health & Fatigue Recommendation System")
st.caption("K-Nearest Neighbours + Random Forest trained on NASA ASRS air-carrier fatigue reports")
st.warning(DISCLAIMER)

tab_assess, tab_perf, tab_data = st.tabs(["Assess my routine", "Model performance", "Dataset insights"])

# ------------------------------------------------------------------ assess
with tab_assess:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Your routine")
        sleep = st.slider("Sleep in the last 24 h (hours)", 0.0, 12.0, 6.0, 0.5)
        duty = st.slider("Hours on duty so far", 0.0, 18.0, 8.0, 0.5)
        legs = st.number_input("Legs flown today", 0, 8, 2)
        flipped = st.checkbox("Day/night schedule flipped recently")
        changed = st.checkbox("Trip was revised / reassigned")
        delay = st.checkbox("Significant delays today")
        commute = st.checkbox("Commuted or deadheaded before duty")
    with c2:
        st.subheader("Operational context")
        tb = st.selectbox("Local time of day", TIME_BUCKETS[:-1],
                          format_func=lambda s: {"night": "00:01-06:00", "morning": "06:01-12:00",
                                                 "afternoon": "12:01-18:00", "evening": "18:01-24:00"}[s])
        light = st.selectbox("Light condition", LIGHT_CONDS[:-1], index=0)
        phase = st.selectbox("Flight phase", PHASE_GROUPS[:-1], index=3)
        role = st.selectbox("Role", ROLES)
        fam = st.selectbox("Aircraft family", AIRCRAFT_FAMILIES)
        wx = st.checkbox("Weather disruption (storms, icing, de-icing)")
        wl = st.checkbox("High workload / time pressure")
        unf = st.checkbox("Unfamiliar airport or aircraft")

    if st.button("Assess", type="primary"):
        profile = profile_from_routine(
            sleep_hours_24h=sleep, hours_on_duty=duty, legs_flown=int(legs), time_bucket=tb,
            light_cond=light, phase_group=phase, role=role, aircraft_family=fam,
            schedule_changed=changed, flipped_day_night=flipped, significant_delay=delay,
            commuted_or_deadheaded=commute, weather_disruption=wx, high_workload=wl,
            unfamiliar_airport_or_aircraft=unf,
        )
        a = rec.assess(profile)

        m1, m2 = st.columns(2)
        m1.metric("Random Forest risk level", a.risk_rf)
        m2.metric("KNN risk level (vote of similar reports)", a.risk_knn)
        st.bar_chart(pd.Series(a.rf_probabilities, name="probability"))

        st.subheader("Recommendations")
        for r in a.recommendations:
            with st.expander(f"{r['title']}  ·  based on {r['source']}", expanded=r["factor"] == "overall"):
                for line in r["advice"]:
                    st.markdown(f"- {line}")

        st.subheader("Most similar historical ASRS reports (KNN)")
        st.dataframe(a.neighbours, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ performance
with tab_perf:
    mp = MODEL_DIR / "metrics.json"
    if not mp.exists():
        st.info("Run `python -m src.train` to generate metrics.")
    else:
        metrics = json.loads(mp.read_text())
        st.write(f"**{metrics['n_records']} reports** · class counts: {metrics['class_counts']}")
        st.caption("Repeated stratified cross-validation with nested hyper-parameter search.")
        rows = [{"model": k, "accuracy": f"{v['accuracy_mean']:.3f} ± {v['accuracy_std']:.3f}",
                 "macro-F1": f"{v['f1_macro_mean']:.3f} ± {v['f1_macro_std']:.3f}"}
                for k, v in metrics["cv"].items()]
        st.table(pd.DataFrame(rows))
        for name, cm in metrics.get("confusion", {}).items():
            st.markdown(f"**{name} - confusion matrix (out-of-fold)**")
            st.dataframe(pd.DataFrame(cm["matrix"], index=[f"true {l}" for l in cm["labels"]],
                                      columns=[f"pred {l}" for l in cm["labels"]]))
        fi = MODEL_DIR / "feature_importance.csv"
        if fi.exists():
            st.markdown("**Random Forest feature importance**")
            st.bar_chart(pd.read_csv(fi).head(15).set_index("feature"))
        st.caption("With so few records, treat scores as indicative only - compare against the majority baseline.")

# ------------------------------------------------------------------ data
with tab_data:
    ref = rec.reference
    st.write(f"{len(ref)} historical reports")
    st.markdown("**How often each routine factor appears in the reports**")
    st.bar_chart(ref[FLAG_COLS].mean().sort_values(ascending=False))
    st.markdown("**Risk level distribution**")
    st.bar_chart(ref["risk_level"].value_counts())
