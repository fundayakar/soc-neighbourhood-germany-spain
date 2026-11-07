"""Attach independently licensed 0–20 cm SOC records to Spanish EO inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def attach(labels, output, id_column, value_column, depth_column, sep):
    output = Path(output).resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Place labelled output outside this repository.")
    if output.exists():
        raise FileExistsError(output)
    meta = json.loads((ROOT / "spain/data/predictor_manifest.json").read_text())
    source = ROOT / meta["path"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != meta["predictors_sha256"]:
        raise ValueError("Spanish predictor checksum mismatch.")
    with np.load(source, allow_pickle=False) as d:
        arrays = dict(d)
    ids = arrays["point_id"]
    frame = pd.read_csv(labels, dtype={id_column: str}, sep=sep, low_memory=False)
    frame[id_column] = frame[id_column].str.strip()
    frame = frame[frame[id_column].isin(ids)]
    if depth_column in frame:
        depth = frame[depth_column].astype(str).str.strip().str.replace("–", "-", regex=False)
        frame = frame[depth.isin(["0-20 cm", "0-20"])]
    if frame[id_column].duplicated().any():
        raise ValueError("Duplicate cohort IDs after the depth filter.")
    target = pd.to_numeric(frame.set_index(id_column)[value_column], errors="raise").reindex(ids)
    values = target.to_numpy(dtype=meta["label_dtype"])
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Missing or invalid SOC values for the frozen cohort.")
    if hashlib.sha256(values.tobytes()).hexdigest() != meta["label_array_sha256"]:
        raise ValueError("SOC values differ from the study's frozen label array.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f:
        np.savez_compressed(f, **arrays, y=values)
    print(f"Attached {len(ids)} independently supplied labels; exact SOC hash matched.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--id-column", default="POINT_ID")
    p.add_argument("--value-column", default="OC")
    p.add_argument("--depth-column", default="Depth")
    p.add_argument("--sep", default=",")
    attach(**vars(p.parse_args()))
