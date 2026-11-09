"""First-pass crop mapping probes on precomputed feature vectors.

This is a baseline screening study, NOT calibrated modality trust or true
per-date temporal attention. Main performance uses spatial blocks with
an optional geographic exclusion buffer.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import BallTree
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


REQUIRED = {"field_id", "lat", "lon", "capture_year", "capture_month", "crop_label"}


def load_data(path):
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    sat_cols = sorted([c for c in df if c.startswith("s_")])
    ground_cols = sorted([c for c in df if c.startswith("g_")])
    if not sat_cols or not ground_cols:
        raise ValueError("Need numerical s_* satellite and g_* ground embeddings")
    if df[list(REQUIRED)].isna().any().any():
        raise ValueError("Required metadata/labels must not contain nulls")
    if not df["capture_month"].between(1, 12).all():
        raise ValueError("capture_month must be 1...12")
    if not df["lat"].between(-90, 90).all() or not df["lon"].between(-180, 180).all():
        raise ValueError("Invalid lat/lon")
    if df["field_id"].duplicated().any():
        raise ValueError("Pilot requires one observation per field; aggregate field repeats upstream")
    for c in sat_cols + ground_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "ground_quality" in df.columns:
        df["ground_quality"] = pd.to_numeric(df["ground_quality"], errors="coerce").clip(0, 1)
    else:
        df["ground_quality"] = 1.0
    return df, sat_cols, ground_cols


def spatial_split(df, block_degrees=1.0, buffer_km=0.0):
    """Deterministic spatial checkerboard: 1/4 blocks are held out.

    A purge removes train samples too close to any test point.
    """
    if block_degrees <= 0 or buffer_km < 0:
        raise ValueError("block_degrees must be positive and buffer_km nonnegative")
    lat_block = np.floor(df.lat.to_numpy() / block_degrees).astype(int)
    lon_block = np.floor(df.lon.to_numpy() / block_degrees).astype(int)
    test = ((lat_block + lon_block) % 4 == 0)
    train = ~test
    if buffer_km > 0 and test.any() and train.any():
        train_points = np.deg2rad(df.loc[train, ["lat", "lon"]].to_numpy())
        test_points = np.deg2rad(df.loc[test, ["lat", "lon"]].to_numpy())
        dist, _ = BallTree(test_points, metric="haversine").query(train_points, k=1)
        keep = (dist[:, 0] * 6371.0088) >= buffer_km
        original_train = np.flatnonzero(train)
        train = np.zeros(len(df), dtype=bool)
        train[original_train[keep]] = True
    if train.sum() < 15 or test.sum() < 8:
        raise ValueError("Spatial split too small. Adjust block size or dataset coverage.")
    return np.flatnonzero(train), np.flatnonzero(test)


def cyclic_month(month):
    angle = 2 * np.pi * (np.asarray(month) - 1) / 12
    return np.column_stack([np.sin(angle), np.cos(angle)])


def features(df, sat_cols, ground_cols, kind, shuffled_month=None):
    sat = df[sat_cols].to_numpy(dtype=float)
    ground = df[ground_cols].to_numpy(dtype=float)
    mm = df.capture_month.to_numpy() if shuffled_month is None else np.asarray(shuffled_month)
    cyc = cyclic_month(mm)
    if kind == "satellite":
        return sat
    if kind == "ground":
        return ground
    if kind == "concat":
        return np.column_stack([sat, ground])
    if kind in ("month", "month_shuffled"):
        return np.column_stack([sat, ground, cyc])
    if kind in ("month_interaction", "quality_interaction"):
        interaction = np.concatenate([ground * cyc[:, i:i+1] for i in range(2)], axis=1)
        out = [sat, ground, cyc, interaction]
        if kind == "quality_interaction":
            q = df.ground_quality.fillna(1).to_numpy(dtype=float)[:, None]
            out.extend([q, ground * q])
        return np.column_stack(out)
    raise ValueError(f"Unknown kind: {kind}")


def fit_model(X, y, train_index):
    model = make_pipeline(SimpleImputer(strategy="constant", fill_value=0),
                          StandardScaler(),
                          LogisticRegression(max_iter=1500, class_weight="balanced"))
    model.fit(X[train_index], y[train_index])
    return model


def score(model, X, y, test_index):
    predicted = model.predict(X[test_index])
    return {"n": int(len(test_index)),
            "accuracy": float(accuracy_score(y[test_index], predicted)),
            "macro_f1": float(f1_score(y[test_index], predicted,
                                       labels=np.unique(y), average="macro",
                                       zero_division=0))}


def run(df, sat_cols, ground_cols, block_degrees=1.0, buffer_km=0.0, seed=7):
    train, test = spatial_split(df, block_degrees, buffer_km)
    y = df.crop_label.astype(str).to_numpy()
    if len(np.unique(y[train])) < 2:
        raise ValueError("Training needs at least two crop classes")
    variants = ["satellite", "ground", "concat", "month", "month_shuffled",
                "month_interaction", "quality_interaction"]
    result = {"protocol": {"spatial_block_degrees": block_degrees,
                            "buffer_km": buffer_km,
                            "train_n": len(train), "test_n": len(test),
                            "seed": seed,
                            "warnings": [
                                "Synthetic and unverified labels are NOT valid scientific evaluation",
                                "Month interactions do not implement temporal sequence alignment",
                                "Quality interaction is not calibrated reliability"
                            ]},
              "class_counts": df.crop_label.value_counts().to_dict(),
              "month_counts": {str(k): int(v) for k, v in
                               df.capture_month.value_counts().sort_index().items()},
              "models": {}, "corruptions": {}}
    rng = np.random.default_rng(seed)
    shuffled = df.capture_month.to_numpy().copy()
    # Independently permute train and test; no labels used in the shuffle.
    shuffled[train] = rng.permutation(shuffled[train])
    shuffled[test] = rng.permutation(shuffled[test])
    for kind in variants:
        X = features(df, sat_cols, ground_cols, kind,
                     shuffled_month=shuffled if kind == "month_shuffled" else None)
        model = fit_model(X, y, train)
        result["models"][kind] = score(model, X, y, test)
        if kind == "concat":
            result["corruptions"]["clean"] = result["models"]["concat"]
            for rate in (0.1, 0.3, 0.5):
                corrupted = df.copy()
                subset = rng.choice(test, size=max(1, round(len(test) * rate)), replace=False)
                # Same-season wrong-field proxy: random ground embeddings from elsewhere.
                donors = rng.choice(test, size=len(subset), replace=True)
                corrupted.loc[subset, ground_cols] = df.iloc[donors][ground_cols].to_numpy()
                X_bad = features(corrupted, sat_cols, ground_cols, "concat")
                result["corruptions"][f"ground_mismatch_{rate}"] = score(model, X_bad, y, test)
            for rate in (0.3, 0.6):
                corrupted = df.copy()
                subset = rng.choice(test, size=max(1, round(len(test) * rate)), replace=False)
                corrupted.loc[subset, sat_cols] = 0
                X_bad = features(corrupted, sat_cols, ground_cols, "concat")
                result["corruptions"][f"satellite_missing_{rate}"] = score(model, X_bad, y, test)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--block-degrees", type=float, default=1.0)
    parser.add_argument("--buffer-km", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    df, sat, ground = load_data(args.input)
    outcome = run(df, sat, ground, args.block_degrees, args.buffer_km, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(outcome, indent=2))
    print(json.dumps(outcome["models"], indent=2))
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
