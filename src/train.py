"""Train and evaluate the KNN and Random Forest models.

    python -m src.train

Outputs (in models/):
    knn.joblib, rf.joblib          fitted sklearn Pipelines
    reference.joblib               feature table + metadata of the historical reports (for neighbours)
    metrics.json                   cross-validated scores (KNN, Random Forest, majority baseline)
    feature_importance.csv         Random Forest feature importances
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src.features import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    FLAG_COLS,
    NUMERIC_COLS,
    RISK_LEVELS,
    add_labels,
    build_features,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "processed" / "asrs_fatigue.csv"
MODEL_DIR = ROOT / "models"
SEED = 42


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", MinMaxScaler(), NUMERIC_COLS),
            ("flags", "passthrough", FLAG_COLS),
        ]
    )


def make_knn(inner_splits: int = 3) -> GridSearchCV:
    pipe = Pipeline([("prep", make_preprocessor()), ("knn", KNeighborsClassifier(weights="distance"))])
    grid = {"knn__n_neighbors": [3, 5, 7], "knn__metric": ["euclidean", "manhattan"]}
    return GridSearchCV(pipe, grid, cv=_inner_cv(inner_splits), scoring="f1_macro")


def make_rf(inner_splits: int = 3) -> GridSearchCV:
    pipe = Pipeline(
        [
            ("prep", make_preprocessor()),
            ("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED)),
        ]
    )
    grid = {"rf__max_depth": [None, 4], "rf__min_samples_leaf": [1, 2]}
    return GridSearchCV(pipe, grid, cv=_inner_cv(inner_splits), scoring="f1_macro")


def _inner_cv(n_splits: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def _safe_splits(y: pd.Series, wanted: int = 5) -> int:
    min_count = int(y.value_counts().min())
    if min_count < 2:
        raise ValueError(
            "At least one risk class has fewer than 2 samples, so stratified cross-validation is "
            f"impossible. Class counts: {y.value_counts().to_dict()}. "
            "Merge classes in features.derive_risk_level or use more data."
        )
    return min(wanted, min_count)


def run(df: pd.DataFrame, model_dir: Path = MODEL_DIR, repeats: int = 5, verbose: bool = True) -> dict:
    df = df.fillna("").reset_index(drop=True)
    X = build_features(df)
    y = add_labels(df)
    n_splits = _safe_splits(y)
    inner = max(2, min(3, n_splits))

    if verbose:
        print(f"Records: {len(df)} | class balance: {y.value_counts().to_dict()}")

    outer = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=repeats, random_state=SEED)
    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}
    estimators = {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "KNN": make_knn(inner),
        "Random Forest": make_rf(inner),
    }

    metrics: dict = {"n_records": int(len(df)), "class_counts": y.value_counts().to_dict(), "cv": {}}
    for name, est in estimators.items():
        res = cross_validate(est, X, y, cv=outer, scoring=scoring, n_jobs=1)
        metrics["cv"][name] = {
            "accuracy_mean": float(np.mean(res["test_accuracy"])),
            "accuracy_std": float(np.std(res["test_accuracy"])),
            "f1_macro_mean": float(np.mean(res["test_f1_macro"])),
            "f1_macro_std": float(np.std(res["test_f1_macro"])),
        }
        if verbose:
            m = metrics["cv"][name]
            print(
                f"{name:<18} accuracy {m['accuracy_mean']:.3f} ± {m['accuracy_std']:.3f} | "
                f"macro-F1 {m['f1_macro_mean']:.3f} ± {m['f1_macro_std']:.3f}"
            )

    # out-of-fold predictions -> confusion matrices / reports (one pass, single split)
    single = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    metrics["confusion"] = {}
    for name in ("KNN", "Random Forest"):
        pred = cross_val_predict(estimators[name], X, y, cv=single)
        labels = [lvl for lvl in RISK_LEVELS if lvl in set(y)]
        metrics["confusion"][name] = {
            "labels": labels,
            "matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        }
        if verbose:
            print(f"\n{name} out-of-fold report")
            print(classification_report(y, pred, labels=labels, zero_division=0))

    # fit final models on all data and persist
    model_dir.mkdir(parents=True, exist_ok=True)
    knn = make_knn(inner).fit(X, y)
    rf = make_rf(inner).fit(X, y)
    metrics["best_params"] = {"KNN": knn.best_params_, "Random Forest": rf.best_params_}

    best_rf = rf.best_estimator_
    names = best_rf.named_steps["prep"].get_feature_names_out()
    imp = pd.DataFrame(
        {
            "feature": [n.split("__", 1)[-1] for n in names],
            "importance": best_rf.named_steps["rf"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    imp.to_csv(model_dir / "feature_importance.csv", index=False)

    joblib.dump(knn.best_estimator_, model_dir / "knn.joblib")
    joblib.dump(best_rf, model_dir / "rf.joblib")
    reference = X.copy()
    reference["risk_level"] = y.values
    for c in ("acn", "synopsis"):
        reference[c] = df[c].values if c in df else ""
    joblib.dump(reference, model_dir / "reference.joblib")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if verbose:
        print("\nTop features (Random Forest):")
        print(imp.head(10).to_string(index=False))
        print(f"\nSaved models to {model_dir}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    if not args.csv.exists():
        raise SystemExit(f"{args.csv} not found. Run `python -m src.parse_asrs` first.")
    run(pd.read_csv(args.csv), repeats=args.repeats)


if __name__ == "__main__":
    main()
