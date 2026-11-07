"""Regenerate the manuscript tables from the archived aggregate CSVs."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DE = "reproducibility/results/"
ES = "spain/results/"
PROTOCOLS = ["block_100", "block_100_buffer20"]
PROTOCOL_LABELS = dict(zip(PROTOCOLS, ["Blocks", "Blocks + buffer"]))
MODELS = {"centerRF": "Centre RF", "contextRF": "Context RF", "centerMLP": "Centre MLP", "patchCNN": "Patch CNN"}
VARIANTS = {"center_tiled": "Centre only", "original_replay": "Original", "shuffle_314159": "Joint shuffle"}
CONTRASTS = {"original_vs_centre": "Original − centre only", "original_vs_shuffled": "Original − joint shuffle"}
SCORE_HEADERS = ["RMSE (g kg⁻¹)", "MAE (g kg⁻¹)", "Log-scale R²"]

def read(p):
    with (ROOT / p).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def index(rows, *keys):
    out = {tuple(r[k] for k in keys): r for r in rows}
    if len(out) != len(rows):
        raise ValueError(f"Duplicate aggregate keys: {keys}")
    return out

def num(value, signed=False, decimals=3):
    return format(float(value), f"{'+' if signed else ''}.{decimals}f").replace("-", "−")

def interval(row, low, high, decimals=3):
    return f"[{num(row[low], decimals=decimals)}, {num(row[high], decimals=decimals)}]"

def estimate(row, value, low, high, decimals=3):
    return f"{num(row[value], True, decimals)} {interval(row, low, high, decimals)}"

def triple(row, german=False):
    suffix = "_mean" if german else ""
    return [num(row[k + suffix]) for k in ["raw_rmse", "raw_mae", "log_r2"]]

def tables():
    de = index(read(DE + "main_summary.csv"), "year", "protocol", "target", "model")
    es = index(read(ES + "full/mean_scores.csv"), "protocol", "target", "model")
    pairs = {reg: index(read(f"{ES}{reg}/paired_contrasts.csv"), "protocol", "target", "contrast")
             for reg in ["full", "reduced"]}
    effect = index(read(DE + "control_effects.csv"), "protocol", "year", "target", "comparison")
    gps = index(read(DE + "gps_summary.csv"), "year", "protocol", "target", "model")
    sizes = index(read(ES + "training_size/training_size_context_interactions.csv"), "protocol", "control")
    h = {reg: index(read(f"{ES}{reg}/heterogeneity_slopes.csv"), "protocol", "contrast")
         for reg in ["full", "reduced"]}

    t2 = [["Country", "Training target", "Model", *SCORE_HEADERS]]
    t3 = [["Country", "Spatial protocol", "Δ log-scale R²: original − centre",
           "Δ log-scale R²: original − shuffled", "Δ RMSE: original − centre", "Δ RMSE: original − shuffled"]]
    for country in ["Germany", "Spain"]:
        for target in ["log", "raw"]:
            for m in ["centerRF", "contextRF", "patchCNN"]:
                model = "Original CNN" if m == "patchCNN" else MODELS[m]
                vals = triple(de["2018", "block_100", target, m], True) if country == "Germany" else triple(es["block_100", target, "original_replay" if m == "patchCNN" else m])
                t2.append([country, target.title(), model, *vals])
        for protocol in PROTOCOLS:
            if country == "Germany":
                a, b = [effect[protocol, "2018", "log", key] for key in ["original−center_tiled", "original−shuffle_314159"]]
                vals = [a["delta_log_R2"], b["delta_log_R2"], a["delta_RMSE"], b["delta_RMSE"]]
            else:
                a, b = [pairs["full"][protocol, "log", key] for key in CONTRASTS]
                vals = [a["log_r2_delta"], b["log_r2_delta"], a["raw_rmse_delta"], b["raw_rmse_delta"]]
            label = "100 km blocks" + (" + 20 km buffer" if protocol != "block_100" else "")
            t3.append([country, label, *[num(v, True) for v in vals]])

    s10 = [["Spatial protocol", "Training target", "Representation", *SCORE_HEADERS]]
    s11 = [["Predictor year", "Spatial protocol", "Training target", "Model", "Nominal RMSE", "GPS RMSE", "ΔRMSE", "95% interval for ΔRMSE"]]
    s12 = [["Spatial protocol", "Training target", "Contrast", "Δ log-scale R² [95% interval]", "Δ RMSE [95% interval]", "Δ MAE [95% interval]"]]
    s13 = [["Spatial protocol", "Contrast", "Reduced training [95% interval]", "Full training [95% interval]", "Change: full − reduced [95% interval]"]]
    s14 = [["Training set", "Spatial protocol", "Contrast", "Slope per IQR increase", "95% interval"]]
    for p in PROTOCOLS:
        label = PROTOCOL_LABELS[p]
        for target in ["log", "raw"]:
            for variant, name in VARIANTS.items():
                s10.append([label, target.title(), name, *triple(es[p, target, variant])])
            for k, name in CONTRASTS.items():
                r = pairs["full"][p, target, k]
                s12.append([label, target.title(), name, *[estimate(r, f"{j}_delta", f"{j}_ci_low", f"{j}_ci_high") for j in ["log_r2", "raw_rmse", "raw_mae"]]])
        for k, control in zip(CONTRASTS, ["center_tiled", "shuffle_314159"]):
            r = sizes[p, control]
            vals = [estimate(pairs[reg][p, "log", k], "log_r2_delta", "log_r2_ci_low", "log_r2_ci_high") for reg in ["reduced", "full"]]
            s13.append([label, CONTRASTS[k], *vals, estimate(r, "log_R2_gain_change", "log_R2_change_ci_low", "log_R2_change_ci_high")])
    for y in ["2018", "2022"]:
        for p in PROTOCOLS:
            for target in ["raw", "log"]:
                for model, name in MODELS.items():
                    r = gps[y, p, target, model]
                    s11.append([y, PROTOCOL_LABELS[p], target.title(), name, num(r["nominal_rmse"]), num(r["gps_rmse"]), num(r["delta_rmse"], True), interval(r, "delta_rmse_ci_low", "delta_rmse_ci_high")])
    for reg in ["full", "reduced"]:
        for p in PROTOCOLS:
            for k, name in CONTRASTS.items():
                r = h[reg][p, k]
                s14.append([reg.title(), PROTOCOL_LABELS[p], name.replace(" − ", " vs "), num(r["slope_per_NDVI_std_IQR"], True, 4), interval(r, "ci_low", "ci_high", 4)])
    return dict(zip(["Table_2", "Table_3", "Table_S10", "Table_S11", "Table_S12", "Table_S13", "Table_S14"],
                    [t2, t3, s10, s11, s12, s13, s14]))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    p = parser.parse_args().output
    p.mkdir(parents=True, exist_ok=True)
    for name, rows in tables().items():
        with (p / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        print(f"{name}: {len(rows) - 1} data rows")

if __name__ == "__main__":
    main()
