#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import base64, io, cv2, pandas as pd, numpy as np, torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# --------------------------------------------------------------------------- #
def _apply_mask(img: np.ndarray, mask: np.ndarray | None):
    if mask is None:            # καμία μάσκα διαθέσιμη
        return img
    mask = (mask > 0.5).astype(np.uint8)
    return cv2.bitwise_and(img, img, mask=mask.squeeze())

def _decode_base64(b64: str) -> np.ndarray:
    buf = base64.b64decode(b64)
    img = Image.open(io.BytesIO(buf)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# --------------------------------------------------------------------------- #
class LungDataset(Dataset):
    """
    Υποστηρίζονται τρία σενάρια:
      1. DicomPath + MaskPath
      2. Μόνο DicomPath (use_mask=False)
      3. Στήλη 'image' (base-64) χωρίς αρχεία δίσκου
    """
    def __init__(self, csv_or_df, label_cols,
                 img_size=224, aug=False, use_mask=True):
        self.df   = pd.read_csv(csv_or_df) if isinstance(csv_or_df, str) else csv_or_df
        self.cols = label_cols
        self.mask = use_mask and "MaskPath" in self.df.columns

        self.tf = A.Compose([
            (A.RandomResizedCrop(img_size, img_size, (.8,1.0))
             if aug else A.Resize(img_size, img_size)),
            A.HorizontalFlip(p=.5)                if aug else A.NoOp(),
            A.RandomBrightnessContrast(.1,.1,p=.3) if aug else A.NoOp(),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2()
        ])

    # --------------------------------------------------------------------- #
    def __len__(self): return len(self.df)

    def _read_image(self, row):
        if "image" in row:                             # base-64
            img = _decode_base64(row.image)
        else:                                          # από δίσκο
            img = cv2.imread(row.DicomPath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(row.DicomPath)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def __getitem__(self, idx):
        r, mask_arr = self.df.iloc[idx], None
        img = self._read_image(r)
        if self.mask:
            mask_arr = cv2.imread(r.MaskPath, cv2.IMREAD_GRAYSCALE)
        img = _apply_mask(img, mask_arr)
        img = self.tf(image=img)["image"]
        y   = torch.tensor(r[self.cols].values.astype("float32"))
        return img, y
