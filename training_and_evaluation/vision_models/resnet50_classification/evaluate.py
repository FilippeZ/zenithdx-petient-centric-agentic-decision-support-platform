#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os, torch, numpy as np, pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from dataset import LungDataset
from model   import build_resnet50
from utils   import set_seed, get_device, json_save

# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device, label_cols, save_json=None, save_proba=None):
    model.eval()
    ys, ps = [], []
    for x, y in tqdm(loader, desc="test"):
        x = x.to(device)
        ps.append(torch.sigmoid(model(x)).cpu().numpy())
        ys.append(y.numpy())
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    pred   = (ps >= 0.5).astype(int)

    metrics = dict(
        accuracy  = float(accuracy_score(ys.reshape(-1), pred.reshape(-1))),
        precision = float(precision_score(ys.reshape(-1), pred.reshape(-1), zero_division=0)),
        recall    = float(recall_score   (ys.reshape(-1), pred.reshape(-1), zero_division=0)),
        f1        = float(f1_score       (ys.reshape(-1), pred.reshape(-1), zero_division=0)),
        roc_auc   = float(roc_auc_score (ys, ps, average="macro")),
        per_class = {},
        confusion = {}
    )
    for i, cls in enumerate(label_cols):
        metrics["per_class"][cls] = dict(
            acc  = accuracy_score (ys[:,i], pred[:,i]),
            prec = precision_score(ys[:,i], pred[:,i], zero_division=0),
            rec  = recall_score   (ys[:,i], pred[:,i], zero_division=0),
            f1   = f1_score       (ys[:,i], pred[:,i], zero_division=0),
            auc  = roc_auc_score  (ys[:,i], ps[:,i])
        )
        metrics["confusion"][cls] = confusion_matrix(ys[:,i], pred[:,i]).tolist()

    if save_json:  json_save(Path(save_json), metrics)
    if save_proba:
        df_out = pd.DataFrame(ps, columns=label_cols)
        df_out.to_csv(save_proba, index=False)
    return metrics
# --------------------------------------------------------------------------- #
def main(cfg):
    set_seed(42); device = get_device()
    ckpt = torch.load(cfg.ckpt, map_location="cpu")
    label_cols = ckpt["label_cols"]

    df = pd.read_csv(os.path.join(cfg.base_dir, cfg.csv))
    ds = LungDataset(df, label_cols, cfg.img_size, aug=False,
                     use_mask=("MaskPath" in df.columns))
    ld = DataLoader(ds, batch_size=cfg.bs, shuffle=False,
                    num_workers=4, pin_memory=device.type=="cuda")

    model = build_resnet50(len(label_cols), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    m = evaluate(model, ld, device, label_cols,
                 save_json=Path(cfg.out_dir)/"test_metrics.json",
                 save_proba=Path(cfg.out_dir)/"proba.csv")
    print(json.dumps(m, indent=2))
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", default="./runs_resnet50")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--bs", type=int, default=64)
    cfg = ap.parse_args(); main(cfg)
