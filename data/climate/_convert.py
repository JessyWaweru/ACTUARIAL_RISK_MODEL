"""One-off conversion of raw NASA POWER / World Bank JSON pulls into clean CSVs.

Run once after re-fetching raw_*.json; not part of the package (data/ is not
importable), kept here only so the provenance of the committed CSVs is clear
and reproducible.
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent

def convert_monthly(name: str) -> None:
    raw = json.loads((HERE / f"rainfall_monthly_{name}_raw.json").read_text())
    series = raw["properties"]["parameter"]["PRECTOTCORR"]
    lat, lon = raw["geometry"]["coordinates"][1], raw["geometry"]["coordinates"][0]
    rows = []
    for key, value in series.items():
        year, month = int(key[:4]), int(key[4:6])
        if month == 13:  # NASA POWER appends a synthetic annual-average "month"
            continue
        if value == -999.0:  # NASA POWER fill value
            continue
        rows.append((year, month, value))
    rows.sort()
    out = HERE / f"rainfall_monthly_{name}.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", "month", "rainfall_mm"])
        w.writerows(rows)
    print(f"{name}: {len(rows)} months -> {out.name} (lat={lat}, lon={lon})")


def convert_daily(name: str) -> None:
    raw = json.loads((HERE / f"rainfall_daily_{name}_raw.json").read_text())
    series = raw["properties"]["parameter"]["PRECTOTCORR"]
    rows = []
    for key, value in series.items():
        year, month, day = int(key[:4]), int(key[4:6]), int(key[6:8])
        if value == -999.0:
            continue
        rows.append((f"{year:04d}-{month:02d}-{day:02d}", value))
    rows.sort()
    out = HERE / f"rainfall_daily_{name}.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "rainfall_mm"])
        w.writerows(rows)
    print(f"{name}: {len(rows)} days -> {out.name}")


def convert_yield() -> None:
    raw_path = HERE.parent / "agriculture" / "kenya_cereal_yield_raw.json"
    raw = json.loads(raw_path.read_text())
    rows = []
    for entry in raw[1]:
        if entry["value"] is not None:
            rows.append((int(entry["date"]), float(entry["value"])))
    rows.sort()
    out = raw_path.parent / "kenya_cereal_yield.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", "yield_kg_per_ha"])
        w.writerows(rows)
    print(f"kenya_cereal_yield: {len(rows)} years -> {out.name}")


if __name__ == "__main__":
    # Garissa only needs the daily series (used for flood/cat-bond EVT), not monthly.
    for county in ("homabay", "westpokot", "turkana"):
        convert_monthly(county)
    convert_daily("garissa")
    convert_yield()
