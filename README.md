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

## Limitations
 
- Fifty reports is a very small dataset, and every one of them is already about fatigue. The accuracy numbers should be read as rough indications, not firm results. Always look at them next to the baseline.
- ASRS reports are voluntary, so the data says nothing about how common any problem really is. NASA says this in the PDF itself.
- The risk label is a rule I wrote, not an expert's judgement. A better version would have aviation safety people label the reports.
- The narrative flags are found with keyword matching, so some will be wrong or missed.
- The advice is general fatigue guidance. It is not medical advice and it does not replace 14 CFR Part 117 or an airline's own fatigue risk management rules.
## Possible next steps
 
Pull a larger set of fatigue reports from the full ASRS database and adapt the parser to its columns, add expert labels, and try text features on the narratives.
