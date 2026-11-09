"""Generate toy embeddings for a smoke test. No real observations or conclusions."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CROPS = ["rice", "cotton", "soybean", "sugarcane", "maize", "millets", "groundnuts", "not cropland"]

def demo(n=600, seed=13):
    rng = np.random.default_rng(seed)
    classes = rng.integers(0, len(CROPS), n)
    month = rng.integers(1, 13, n)
    lat = rng.uniform(10, 27, n)
    lon = rng.uniform(72, 88, n)
    base = rng.normal(size=(len(CROPS), 8))
    sat = base[classes] + rng.normal(scale=.9, size=(n, 8))
    ground = base[classes] + rng.normal(scale=1.1, size=(n, 8))
    mismatch = rng.random(n) < .1
    ground[mismatch] = ground[rng.permutation(n)[:mismatch.sum()]]
    quality = rng.uniform(.3, 1, n)
    df = pd.DataFrame({"field_id": [f"field_{i}" for i in range(n)],
                       "lat": lat, "lon": lon, "capture_year": 2022,
                       "capture_month": month, "crop_label": [CROPS[c] for c in classes],
                       "ground_quality": quality})
    for j in range(8):
        df[f"s_{j}"] = sat[:, j]
        df[f"g_{j}"] = ground[:, j]
    return df

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, default=600)
    args = p.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    demo(args.n).to_csv(args.output, index=False)
    print(f"Wrote synthetic demo: {args.output}")
