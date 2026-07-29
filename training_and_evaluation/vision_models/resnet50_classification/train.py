#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, re, torch, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score

from dataset import LungDataset
from model   import build_resnet50
from utils   import set_seed, get_device, weighted_bce, json_save, metric_dict
# --------------------------------------------------------------------------- #
def epoch_loop(model, loader, crit, opt, device, train=True):
    model.train() if train else model.eval()
    y_true, y_prob, epoch_loss, n = [], [], 0.0, 0
    for x, y in tqdm(loader, leave=False, desc="train" if train else "val  "):
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logit = model(x)
            loss  = crit(logit, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item() * len(x); n += len(x)
        y_true.append(y.cpu().numpy()); y_prob.append(torch.sigmoid(logit).cpu().numpy())
    y_true, y_prob = np.concatenate(y_true), np.concatenate(y_prob)
    m = metric_dict(y_true, y_prob)
    return epoch_loss / n, m["f1"]
# --------------------------------------------------------------------------- #
def build_dataframe(base_dir, csv_file, mask_csv=None):
    import pandas as pd
    df = pd.read_csv(os.path.join(base_dir, csv_file))
    if mask_csv and "MaskPath" not in df.columns:
        # merge μάσκες αν χρειάζεται
        df_m = pd.read_csv(os.path.join(base_dir, mask_csv))
        strip = lambda s: re.sub(r"\.(png|jpg|jpeg)$", "", s)
        df["ID"]  = df.DicomPath.apply(strip)
        df_m["MID"] = df_m.DicomPath.apply(strip)
        df = df.merge(df_m, how="left",
                      left_on="ID",
                      right_on=df_m.MID.str.replace("-mask","", regex=False)) \
               .drop(columns=["ID", "MID"]) \
               .rename(columns={"DicomPath_x":"DicomPath",
                                "DicomPath_y":"MaskPath"})
    df.DicomPath = base_dir + "/" + df.DicomPath
    if "MaskPath" in df.columns:
        df.MaskPath  = base_dir + "/" + df.MaskPath
    return df
# --------------------------------------------------------------------------- #
def main(cfg):
    set_seed(cfg.seed); device = get_device()
    out = Path(cfg.out_dir); (out/"models").mkdir(parents=True, exist_ok=True)

    # ----------------- data ------------------------------------------------ #
    df_tr = build_dataframe(cfg.base_dir, cfg.train_csv, cfg.mask_csv)
    df_vl = build_dataframe(cfg.base_dir, cfg.val_csv,   cfg.mask_csv)
    label_cols = [c for c in df_tr.columns if c not in ["DicomPath","MaskPath","image"]]

    ds_tr = LungDataset(df_tr, label_cols, cfg.img_size, aug=cfg.aug,
                        use_mask=("MaskPath" in df_tr.columns))
    ds_vl = LungDataset(df_vl, label_cols, cfg.img_size, aug=False,
                        use_mask=("MaskPath" in df_vl.columns))

    # optional WeightedRandomSampler (helps on heavy imbalance)
    sampler = None
    if cfg.weighted_sampler:
        lbl_arr = df_tr[label_cols].values
        sample_w = 1. / (lbl_arr.sum(axis=0) + 1e-6)
        weights  = (lbl_arr * sample_w).sum(1)
        sampler  = WeightedRandomSampler(weights, len(weights), replacement=True)

    ld_tr = DataLoader(ds_tr, batch_size=cfg.bs, shuffle=(sampler is None),
                       sampler=sampler, num_workers=4, pin_memory=device.type=="cuda")
    ld_vl = DataLoader(ds_vl, batch_size=cfg.bs, shuffle=False,
                       num_workers=4, pin_memory=device.type=="cuda")

    # ----------------- model ---------------------------------------------- #
    model = build_resnet50(len(label_cols),
                           pretrained=not cfg.no_pretrain,
                           freeze_backbone=cfg.freeze_backbone).to(device)
    crit  = weighted_bce(label_cols, df_tr, device)
    opt   = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=cfg.lr, weight_decay=1e-4)
    sch   = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=2, factor=0.5)

    # ----------------- training ------------------------------------------- #
    best_f1, patience = 0.0, 0
    history = []
    for epoch in range(cfg.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.epochs}")
        tr_loss, tr_f1 = epoch_loop(model, ld_tr, crit, opt, device, True)
        vl_loss, vl_f1 = epoch_loop(model, ld_vl, crit, opt, device, False)
        sch.step(vl_loss)

        history.append(dict(ep=epoch, tr_loss=tr_loss, vl_loss=vl_loss,
                            tr_f1=tr_f1, vl_f1=vl_f1))
        print(f"train {tr_loss:.4f}/{tr_f1:.4f} | val {vl_loss:.4f}/{vl_f1:.4f}")

        if vl_f1 > best_f1 + 1e-4:
            best_f1, patience = vl_f1, 0
            torch.save({"epoch":epoch,
                        "model_state_dict":model.state_dict(),
                        "label_cols":label_cols},
                       out/"models/best_model.pth")
            print(f"✅  new best F1 = {best_f1:.4f}")
        else:
            patience += 1
            if patience >= cfg.patience:
                print("⏹️  early stopping"); break

    json_save(out/"train_summary.json", {"best_f1":best_f1, "history":history})
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir",   required=True)
    ap.add_argument("--train_csv",  required=True)
    ap.add_argument("--val_csv",    required=True)
    ap.add_argument("--mask_csv",   default=None)
    ap.add_argument("--out_dir",    default="./runs_resnet50")
    ap.add_argument("--img_size",   type=int, default=224)
    ap.add_argument("--bs",         type=int, default=32)
    ap.add_argument("--lr",         type=float, default=3e-4)
    ap.add_argument("--epochs",     type=int, default=50)
    ap.add_argument("--patience",   type=int, default=7)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--aug",        action="store_true")
    ap.add_argument("--no_pretrain", action="store_true")
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--weighted_sampler", action="store_true")
    cfg = ap.parse_args(); main(cfg)
