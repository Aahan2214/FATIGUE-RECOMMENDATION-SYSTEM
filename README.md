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

## Quick start
```bash
pip install -r requirements.txt
python scripts/download_data.py        # downloads the NASA PDF
python -m src.parse_asrs               # -> data/processed/asrs_fatigue.csv (expect 50 records)
python -m src.train                    # trains, prints CV results, saves models/
streamlit run app/streamlit_app.py
pytest -q
```

## Deploy
* **GitHub**: `git init && git add . && git commit -m "init" && git remote add origin <url> && git push -u origin main`
  (commit `data/processed/asrs_fatigue.csv` after step 2 so cloud deployments can train automatically).
* **Streamlit Community Cloud** (free): connect the repo, main file `app/streamlit_app.py`.
* **Docker / Render / Hugging Face Spaces (Docker)**: `docker build -t fatigue-app . && docker run -p 8501:8501 fatigue-app`

## Important limitations (put these in your paper)
* Only **50 voluntary reports**; every report is already fatigue-related, so results are indicative, not conclusive.
* The risk label is a **proxy** built by rules; validate with expert labelling.
* ASRS data cannot measure prevalence (NASA caveat). Keyword flags from narratives are approximate.
* To scale up, export more fatigue reports from the full ASRS database and adapt `parse_asrs.py` to its columns.
* Educational tool only; not medical or regulatory advice (see 14 CFR Part 117 and your operator's FRMS).

## Structure
```
src/ (parse_asrs, features, train, recommend)   app/streamlit_app.py   scripts/download_data.py
tests/   data/{raw,processed}   models/   Dockerfile   .github/workflows/ci.yml
```
Data: NASA ASRS (public). Code: MIT.
