# Routine-Based Health and Fatigue Recommendation Systems in Aviation

Machine-learning project using **K-Nearest Neighbours (KNN)** and **Random Forest (RF)** on the
NASA ASRS *Air Carrier (FAR 121) Flight Crew Fatigue Reports* set
(<https://asrs.arc.nasa.gov/docs/rpsts/acr_fatg.pdf>, 50 reports).

## What it does
1. **Parse** the NASA PDF into a table (`src/parse_asrs.py`).
2. **Extract features** describing the crew member's routine: schedule revisions, long duty, delays,
   sleep/rest problems, circadian shift, commuting, weather disruption, workload, unfamiliarity,
   plus time of day, light, flight phase, role, aircraft family (`src/features.py`).
3. **Target**: operational-consequence risk level (Low/Medium/High) derived *only* from the report's
   Anomaly/Result fields, which are never used as inputs (no label leakage).
4. **Models**: KNN and RF, nested grid search, repeated stratified CV, majority-class baseline (`src/train.py`).
5. **Recommender** (`src/recommend.py`): RF gives risk + probabilities; KNN retrieves the most similar
   past reports; countermeasures are ranked from the user's own routine plus factors common in the neighbours.
6. **Web app** (Streamlit) to enter a routine and get recommendations.
