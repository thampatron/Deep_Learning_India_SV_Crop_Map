"""Shared inputs, spatial evaluation and probability metrics for the research probes.

Data rows in metadata.csv must correspond exactly to rows in sequences.npz.
No test or validation targets are used to choose hyperparameters or training features.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss, brier_score_loss
from sklearn.neighbors import BallTree

REQUIRED = {"field_id", "lat", "lon", "capture_year", "capture_month",
            "crop_label", "label_source"}
RADIUS_KM = 6371.0088


def load_inputs(csv_path, npz_path):
    meta = pd.read_csv(csv_path)
    missing = REQUIRED - set(meta)
    if missing:
        raise ValueError(f"Missing metadata: {sorted(missing)}")
    if meta.field_id.duplicated().any():
        raise ValueError("One row per field/season is required; aggregate repeats upstream.")
    if meta[list(REQUIRED)].isna().any().any():
        raise ValueError("Metadata labels and dates cannot be null.")
    if not meta.capture_month.between(1, 12).all():
        raise ValueError("Capture month must be in 1..12.")
    if not meta.lat.between(-90, 90).all() or not meta.lon.between(-180, 180).all():
        raise ValueError("Invalid field coordinates.")
    if not set(meta.label_source.unique()).issubset({"expert", "vlm"}):
        raise ValueError("label_source must be expert or vlm.")
    gcols = sorted(c for c in meta if c.startswith("g_"))
    if not gcols:
        raise ValueError("Need ground embedding columns g_*.")
    g = meta[gcols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(g).all():
        raise ValueError("Ground embeddings must be finite.")
    if "ground_quality" not in meta:
        meta["ground_quality"] = 1.0
    meta["ground_quality"] = pd.to_numeric(meta.ground_quality, errors="coerce")
    if not meta.ground_quality.between(0, 1).all():
        raise ValueError("ground_quality must be in [0,1] and ex-ante.")
    with np.load(npz_path, allow_pickle=False) as d:
        sat = np.asarray(d["satellite"], dtype=float)
        observed = np.asarray(d["observed"], dtype=bool)
        yearmonth = np.asarray(d["yearmonth"], dtype=int)
    if sat.ndim != 3 or observed.shape != sat.shape[:2] or yearmonth.shape != sat.shape[:2]:
        raise ValueError("satellite [N,T,D], observed [N,T], yearmonth [N,T] expected")
    if sat.shape[0] != len(meta):
        raise ValueError("CSV and NPZ row counts must match, in precisely the same order.")
    if not np.isfinite(sat[observed]).all():
        raise ValueError("Observed satellite values must be finite.")
    if not ((yearmonth % 100 >= 1) & (yearmonth % 100 <= 12)).all():
        raise ValueError("Satellite yearmonth must be valid YYYYMM.")
    sat = np.where(observed[..., None], sat, 0.)
    return meta, g, sat, observed, yearmonth


def block_ids(meta, degrees):
    if degrees <= 0:
        raise ValueError("Block size must be positive")
    a = np.floor(meta.lat.to_numpy() / degrees).astype(int)
    b = np.floor(meta.lon.to_numpy() / degrees).astype(int)
    return np.array([f"{x}:{y}" for x, y in zip(a, b)])


def spatial_partition(meta, block_degrees=1., buffer_km=0., seed=13):
    """Assign whole spatial blocks to 60/20/20 train/validation/test; purge neighbors.

    Do not report this single partition as a spatial generalization confidence interval;
    rerun across preregistered block scales and seeds.
    """
    groups = block_ids(meta, block_degrees)
    unique = np.unique(groups)
    if len(unique) < 8:
        raise ValueError("At least 8 spatial blocks required.")
    order = np.random.default_rng(seed).permutation(unique)
    n_test = max(1, int(round(.2 * len(order))))
    n_val = max(1, int(round(.2 * len(order))))
    test_groups = set(order[:n_test])
    val_groups = set(order[n_test:n_test+n_val])
    is_test = np.array([x in test_groups for x in groups])
    is_val = np.array([x in val_groups for x in groups])
    is_train = ~(is_test | is_val)
    if buffer_km:
        if buffer_km < 0:
            raise ValueError("buffer_km must be nonnegative")
        # Purge training locations close to validation OR test locations.
        # Purge validation locations close to test to prevent tuning leakage.
        def distant(from_mask, target_mask):
            if not np.any(target_mask):
                return from_mask
            tree = BallTree(np.deg2rad(meta.loc[target_mask, ["lat","lon"]]),
                            metric="haversine")
            distance, _ = tree.query(np.deg2rad(meta.loc[from_mask, ["lat","lon"]]), k=1)
            original = np.flatnonzero(from_mask)
            good = np.zeros(len(meta), dtype=bool)
            good[original[distance[:,0] * RADIUS_KM >= buffer_km]] = True
            return good
        is_train = distant(is_train, is_val | is_test)
        is_val = distant(is_val, is_test)
    out = {k: np.flatnonzero(v) for k,v in
           [("train",is_train),("val",is_val),("test",is_test)]}
    if any(len(i) < 10 for i in out.values()):
        raise ValueError("Too few rows after spatial split/purge; adjust scale or sample.")
    # Independent expert-only evaluation; never score pseudo-labels as truth.
    for key in ("val","test"):
        ix = out[key]
        out[key] = ix[meta.label_source.to_numpy()[ix] == "expert"]
        if len(out[key]) < 8:
            raise ValueError(f"Need >=8 independent expert-labelled rows in {key}")
    return out


def month_distance(ym, target):
    """Integer month differences; no fictional capture day."""
    return (ym // 100 - target // 100) * 12 + (ym % 100 - target % 100)


def seasonal_features(sat, obs):
    counts = np.maximum(obs.sum(axis=1, keepdims=True), 1)
    mean = sat.sum(axis=1) / counts
    # fixed temporal bins preserve ordering, unlike pooled seasonal embeddings
    flattened = sat.reshape(len(sat), -1)
    masks = obs.astype(float)
    return np.column_stack([mean, flattened, masks])


def temporal_features(meta, ground, sat, observed, yearmonth, variant):
    """A deliberately simple interval-aware local statistic, not cross-attention."""
    season = seasonal_features(sat, observed)
    if variant == "satellite":
        return season
    if variant == "ground":
        return ground
    if variant == "concat":
        return np.column_stack([season, ground])
    capture = meta.capture_year.to_numpy() * 100 + meta.capture_month.to_numpy()
    theta = 2 * np.pi * (meta.capture_month.to_numpy() - 1) / 12
    cyc = np.column_stack([np.sin(theta), np.cos(theta)])
    if variant == "satellite_plus_date":
        return np.column_stack([season, cyc])
    if variant == "concat_plus_date":
        return np.column_stack([season, ground, cyc])
    if variant in ("aligned", "aligned_no_date"):
        # soft month-interval kernel; +/- 1 month is weighted more than far dates
        offsets = month_distance(yearmonth, capture[:,None])
        weights = np.exp(-.5 * (offsets / 1.5) ** 2) * observed
        denom = np.maximum(weights.sum(1, keepdims=True), 1e-8)
        local = (sat * weights[...,None]).sum(1) / denom
        # no observation near anchor: local=0 and coverage=0
        coverage = (weights.sum(1,keepdims=True) > 0).astype(float)
        base = [season, ground, local, coverage, ground * ground.mean(axis=1,keepdims=True)]
        if variant == "aligned":
            base.append(cyc)
        return np.column_stack(base)
    raise ValueError(f"Unknown variant {variant}")


def classifier():
    return make_pipeline(SimpleImputer(strategy="constant", fill_value=0),
                         StandardScaler(),
                         LogisticRegression(max_iter=1200, class_weight="balanced"))


def probabilities(model, x, labels):
    partial = model.predict_proba(x)
    out = np.full((len(x), len(labels)), 1e-8)
    for j, label in enumerate(model.classes_):
        out[:, list(labels).index(label)] = partial[:,j]
    return out / out.sum(axis=1, keepdims=True)


def prediction_metrics(y, prob, labels):
    idx = np.argmax(prob, axis=1)
    pred = np.asarray(labels)[idx]
    onehot = (np.asarray(y)[:,None] == np.asarray(labels)[None,:]).astype(float)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y,pred)),
        "macro_f1": float(f1_score(y,pred,labels=labels,average="macro",zero_division=0)),
        "nll": float(log_loss(y,prob,labels=labels)),
        "multiclass_brier": float(np.mean(np.sum((prob-onehot)**2,axis=1))),
        "per_class_f1": {str(label): float(f1_score(y == label,pred == label,zero_division=0))
                         for label in labels}
    }


def confidence_features(probs, obs, quality, month):
    eps = 1e-8
    entropy = -np.sum(probs * np.log(np.clip(probs,eps,1)),axis=1)
    sorted_p = np.sort(probs,axis=1)
    margin = sorted_p[:,-1] - sorted_p[:,-2] if probs.shape[1] > 1 else sorted_p[:,-1]
    theta = 2*np.pi*(month-1)/12
    return np.column_stack([probs.max(1),entropy,margin,
                            obs.mean(1),quality,np.sin(theta),np.cos(theta)])
