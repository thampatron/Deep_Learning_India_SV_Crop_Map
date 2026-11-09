"""Create a synthetic *software smoke test*, NOT a scientific dataset or result.

Generates expert/VLM labels, per-month satellite observations, sparse ground
representations, acquisition month, 10% wrong-field pairing and cloud gaps.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CROPS = ["rice","cotton","soybean","sugarcane","maize",
         "millets","groundnuts","not cropland"]


def generate(n=1100, seed=73):
    rng = np.random.default_rng(seed)
    # scattered 12x12 spatial blocks: enough for purged spatial CV
    block_x = rng.integers(0,12,n)
    block_y = rng.integers(0,12,n)
    lat = 12 + 1.1 * block_x + rng.uniform(.1,.7,n)
    lon = 73 + 1.1 * block_y + rng.uniform(.1,.7,n)
    crop = rng.integers(0,len(CROPS),n)
    month = rng.integers(1,11,n)
    expert = rng.random(n) < .48
    quality = rng.uniform(.25,1,n)
    # Ground observation depends on crop, visibility, and seasonal state.
    proto = rng.normal(size=(len(CROPS),6))
    ground = proto[crop] + rng.normal(scale=(1.5-quality)[:,None],
                                     size=(n,6))
    # Wrong-field association only disturbs the observed ground representation.
    wrong_match = rng.random(n) < .1
    sources = rng.integers(0,n,wrong_match.sum())
    ground[wrong_match] = ground[sources]
    # VLM label is an imperfect observation of ground, not gold truth.
    confusion_rate = np.clip(.5 - .35*quality + .10*(month <= 2),.05,.8)
    vlm_crop = crop.copy()
    wrong_vlm = rng.random(n) < confusion_rate
    vlm_crop[wrong_vlm] = rng.integers(0,len(CROPS),wrong_vlm.sum())
    # 10 month bins (YYYYMM) and 4 spectral/feature channels.
    months = np.arange(1,11)
    yearmonth = np.broadcast_to(202300 + months,(n,len(months))).copy()
    observed = rng.random((n,len(months))) < .8
    time = np.arange(len(months))
    peaks = np.array([6,5,4,7,5,3,4,0])[crop]
    phenology = np.exp(-.5*((time[None,:]-peaks[:,None])/1.9)**2)
    sat = np.empty((n,len(months),4))
    sat[:,:,0] = phenology + rng.normal(scale=.16,size=(n,len(months)))
    sat[:,:,1] = (crop[:,None] % 4)*.22 + phenology*.4 + rng.normal(scale=.2,size=(n,len(months)))
    sat[:,:,2] = np.sin((time[None,:]+crop[:,None])*.65) + rng.normal(scale=.2,size=(n,len(months)))
    sat[:,:,3] = .3*crop[:,None] + rng.normal(scale=.3,size=(n,len(months)))
    sat[~observed] = np.nan
    meta = pd.DataFrame({
        "field_id":[f"field_{i}" for i in range(n)],
        "lat":lat,"lon":lon,
        "capture_year":2023,"capture_month":month,
        "crop_label":[CROPS[j] for j in np.where(expert,crop,vlm_crop)],
        "label_source":np.where(expert,"expert","vlm"),
        "vlm_label":[CROPS[j] for j in vlm_crop],
        "ground_quality":quality,
        "image_field_match_audited":np.where(expert, (~wrong_match).astype(float), np.nan)
    })
    for j in range(ground.shape[1]):
        meta[f"g_{j}"] = ground[:,j]
    return meta,sat,observed,yearmonth


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",required=True)
    p.add_argument("--n",type=int,default=1100)
    p.add_argument("--seed",type=int,default=73)
    args=p.parse_args()
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    meta,sat,observed,ym=generate(args.n,args.seed)
    meta.to_csv(out/"metadata.csv",index=False)
    np.savez_compressed(out/"sequences.npz",satellite=sat,
                        observed=observed,yearmonth=ym)
    print(f"Created SYNTHETIC smoke-test data under {out}")


if __name__=="__main__":
    main()
