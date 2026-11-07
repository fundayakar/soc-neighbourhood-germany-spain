"""Check the release manifest; the source snapshot can omit the predictor arrays."""
import argparse
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--source-only", action="store_true")
a = p.parse_args()
records = json.loads((root / "MANIFEST.json").read_text())["files"]
checked = 0
for row in records:
    target = root / row["path"]
    if a.source_only and row["predictor_array"] and not target.exists():
        continue
    if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != row["sha256"]:
        raise SystemExit(f"Missing or changed release file: {row['path']}")
    checked += 1
print(f"Verified {checked} release files.")
