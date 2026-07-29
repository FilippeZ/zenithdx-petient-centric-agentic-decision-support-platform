#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pre-process CXLSeg CSV: καθαρισμός – balancing – (προαιρετικά) ενσωμάτωση εικόνων.

Example
-------
python preprocess_csv.py \
    --raw_csv   CXLSeg-segmented.csv \
    --out_csv   CXLSeg-train-updated.csv \
    --img_root  /data/CXLSeg/files \
    --balance_delta 2000
"""
import argparse, os, base64, cv2, pandas as pd, numpy as np
from tqdm import tqdm
# --------------------------------------------------------------------------- #
def encode_image(path, size=(224,224)):
    im = cv2.imread(path)
    if im is None:
        raise FileNotFoundError(path)
    im = cv2.resize(im, size)
    _, buf = cv2.imencode(".jpg", im)
    return base64.b64encode(buf).decode()

def main(cfg):
    df = pd.read_csv(cfg.raw_csv)
    label_cols = [c for c in df.columns if c.lower() not in ["dicompath","maskpath"]]

    # 1. drop -1 / NaN → treat as uncertainty
    df = df.replace(-1, np.nan).dropna(subset=label_cols).fillna(0)

    # 2. class balancing
    minority = int(df[label_cols].sum().min())
    target   = minority + cfg.balance_delta
    keep = set()
    for c in label_cols:
        ids = df[df[c]==1].index.tolist()
        keep.update(np.random.choice(ids, min(len(ids), target), replace=False))
    df = df.loc[list(keep)].reset_index(drop=True)

    # 3. optional base64 embed
    if cfg.img_root:
        tqdm.pandas(desc="encoding images")
        df["image"] = df["DicomPath"].progress_apply(
            lambda p: encode_image(os.path.join(cfg.img_root, p)))
        cols = ["image"] + label_cols
    else:
        cols = ["DicomPath"] + label_cols

    df.to_csv(cfg.out_csv, index=False, columns=cols)
    print(f"✅  saved {len(df)} rows ➜ {cfg.out_csv}")
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_csv",    required=True)
    p.add_argument("--out_csv",    required=True)
    p.add_argument("--img_root",   default=None,
                   help="folder with images – leave blank to skip base64")
    p.add_argument("--balance_delta", type=int, default=2000,
                   help="extra samples allowed over minority class")
    cfg = p.parse_args(); main(cfg)
