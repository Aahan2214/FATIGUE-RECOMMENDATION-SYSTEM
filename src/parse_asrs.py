"""Parse a NASA ASRS "Report Set" PDF (or its extracted text) into a flat CSV.

Usage
-----
    python scripts/download_data.py
    python -m src.parse_asrs                       # data/raw/acr_fatg.pdf -> data/processed/asrs_fatigue.csv
    python -m src.parse_asrs --input my.pdf --output out.csv
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "data" / "raw" / "acr_fatg.pdf"
DEFAULT_CSV = ROOT / "data" / "processed" / "asrs_fatigue.csv"

RECORD_RE = re.compile(r"^ACN:\s*(\d+)\s*\((\d+) of (\d+)\)\s*$", re.M)
FIELD_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 /\.\-]*?)\s:\s?(.*)$")


def read_text(path: Path) -> str:
    """Return the full text of a PDF or a plain-text file."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8")


def _split_records(text: str) -> list[tuple[str, str]]:
    """Split the 'Report Narratives' part of the report set into (ACN, block) pairs."""
    if "Report Narratives" not in text:
        raise ValueError("Could not find the 'Report Narratives' section in the input.")
    body = text.split("Report Narratives", 1)[1]
    matches = list(RECORD_RE.finditer(body))
    records = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        records.append((m.group(1), body[m.end():end]))
    return records


def _parse_block(acn: str, block: str) -> dict:
    split_at = block.find("Narrative: 1")
    header, rest = (block, "") if split_at < 0 else (block[:split_at], block[split_at:])

    fields: dict[str, list[str]] = defaultdict(list)
    for line in header.splitlines():
        m = FIELD_RE.match(line.strip())
        if m:
            fields[m.group(1).strip()].append(m.group(2).strip())

    synopsis = ""
    syn_at = rest.rfind("\nSynopsis")
    if syn_at >= 0:
        synopsis = " ".join(rest[syn_at + len("\nSynopsis"):].split())
        rest = rest[:syn_at]
    narrative = " ".join(re.sub(r"Narrative:\s*\d+", " ", rest).split())

    def first(key: str) -> str:
        return fields[key][0] if fields.get(key) else ""

    def joined(keys: list[str]) -> str:
        vals: list[str] = []
        for k in keys:
            vals += fields.get(k, [])
        return "|".join(dict.fromkeys(v for v in vals if v))

    anomaly_keys = [k for k in fields if k.startswith("Anomaly.")]
    result_keys = [k for k in fields if k.startswith("Result.")]

    return {
        "acn": acn,
        "date": first("Date"),
        "local_time": first("Local Time Of Day"),
        "flight_conditions": first("Flight Conditions"),
        "light": first("Light"),
        "aircraft": first("Make Model Name"),
        "flight_phase": first("Flight Phase"),
        "mission": first("Mission"),
        "crew_size": first("Crew Size.Number Of Crew"),
        "functions": joined(["Function.Flight Crew", "Function.Dispatch"]),
        "human_factors": joined(["Human Factors"]),
        "anomalies": joined(anomaly_keys),
        "anomaly_detail": "|".join(
            f"{k.replace('Anomaly.', '')}: {v}" for k in anomaly_keys for v in fields[k]
        ),
        "results": joined(result_keys),
        "primary_problem": first("Primary Problem"),
        "contributing_factors": joined(["Contributing Factors / Situations"]),
        "synopsis": synopsis,
        "narrative": narrative,
    }


def parse_report_set(text: str) -> pd.DataFrame:
    rows = [_parse_block(acn, block) for acn, block in _split_records(text)]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_PDF)
    ap.add_argument("--output", type=Path, default=DEFAULT_CSV)
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"{args.input} not found. Run `python scripts/download_data.py` first.")

    df = parse_report_set(read_text(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Parsed {len(df)} records -> {args.output}")
    if len(df) != 50:
        print("WARNING: the NASA fatigue report set normally has 50 records - check the CSV.")
    empty = {c: int((df[c] == '').sum()) for c in ("narrative", "human_factors", "flight_phase")}
    print("Empty-field counts:", empty)


if __name__ == "__main__":
    main()
