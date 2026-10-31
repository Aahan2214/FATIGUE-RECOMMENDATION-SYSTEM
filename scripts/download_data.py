"""Download the NASA ASRS fatigue report set (PDF) into data/raw/."""
from pathlib import Path
import requests

URL = "https://asrs.arc.nasa.gov/docs/rpsts/acr_fatg.pdf"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "acr_fatg.pdf"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    DEST.write_bytes(r.content)
    print(f"Saved {len(r.content) / 1024:.0f} KB to {DEST}")


if __name__ == "__main__":
    main()
