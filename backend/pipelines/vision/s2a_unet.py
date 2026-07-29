# backend/pipelines/vision/s2a_unet.py
from __future__ import annotations

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

@keras.utils.register_keras_serializable(package="Custom")
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.concat = layers.Concatenate(axis=-1)
        self.conv = layers.Conv2D(1, kernel_size=kernel_size, padding="same", activation="sigmoid")
        self.multiply = layers.Multiply()

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = self.concat([avg_pool, max_pool])
        attention = self.conv(concat)
        return self.multiply([inputs, attention])

def dice_coef(y_true, y_pred, smooth=1):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(tf.round(y_pred), [-1])
    inter = tf.reduce_sum(y_true * y_pred)
    return (2. * inter + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def load_sa_unet(weights_path: str):
    """
    Load SA-UNet model. Supports:
      - TF SavedModel directory (e.g. data/image/sa_unet_savedmodel/)
      - Legacy Keras .h5 file
    """
    import pathlib
    p = pathlib.Path(weights_path)
    print(f"[SA-UNet] Loading SA-UNet model from: {p}")

    if not p.exists():
        print(f"[SA-UNet] ⚠️  Weights not found at {p}", file=sys.stderr)
        return None

    keras.backend.clear_session()
    custom_objects = {"SpatialAttention": SpatialAttention, "dice_coef": dice_coef}

    try:
        if p.is_dir():
            # TensorFlow SavedModel directory format
            print(f"[SA-UNet] Detected SavedModel directory format", file=sys.stderr)
            model = tf.saved_model.load(str(p))
            # Wrap as a callable for predict-like usage
            print(f"[SA-UNet] ✅ Loaded TF SavedModel from {p}", file=sys.stderr)
            return model
        else:
            # Legacy .h5 or .keras file
            model = keras.models.load_model(
                str(p),
                custom_objects=custom_objects,
                compile=False
            )
            print(f"[SA-UNet] ✅ Loaded Keras model from {p}", file=sys.stderr)
            return model
    except Exception as e:
        print(f"[SA-UNet] ❌ Failed to load model from {p}: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        return None

def sa_unet_predict(model, img_rgb: np.ndarray, target_size=(256, 256)) -> np.ndarray:
    """Predicts lung mask. Works with both Keras model and raw TF SavedModel."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, target_size)
    gray = gray.astype("float32") / 255.0
    batch = gray[None, ..., None]  # (1, H, W, 1)

    try:
        # Try Keras model .predict() first
        if hasattr(model, "predict"):
            pred = model.predict(batch, verbose=0)[0, ..., 0]
        else:
            # TF SavedModel — use serving_default or __call__
            infer_fn = None
            if hasattr(model, "signatures"):
                sig = model.signatures.get("serving_default") or next(iter(model.signatures.values()), None)
                if sig:
                    input_key = list(sig.structured_input_signature[1].keys())[0]
                    output_key = list(sig.structured_outputs.keys())[0]
                    result = sig(**{input_key: tf.constant(batch)})
                    pred = result[output_key].numpy()[0, ..., 0]
                    infer_fn = True
            if not infer_fn:
                # Direct __call__ as fallback
                pred = model(tf.constant(batch), training=False).numpy()[0, ..., 0]
    except Exception as e:
        print(f"[SA-UNet] Predict error: {e}", file=sys.stderr)
        # Return a full mask (no segmentation) as safe fallback
        return np.ones((img_rgb.shape[0], img_rgb.shape[1]), dtype="float32")

    bin_mask = (pred > 0.5).astype("float32")
    full_mask = cv2.resize(
        bin_mask,
        (img_rgb.shape[1], img_rgb.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    return full_mask

def apply_mask(original_rgb: np.ndarray, mask: np.ndarray):
    """Returns (masked_lung_image, blended_overlay_with_green_lung) from original RGB and mask."""
    lung = (original_rgb * mask[..., None]).astype("uint8")
    overlay = original_rgb.copy()
    green_mask = np.zeros_like(original_rgb)
    green_mask[..., 1] = 255
    overlay = np.where(mask[..., None] > 0.5, green_mask, overlay)
    alpha = 0.35
    blended = cv2.addWeighted(original_rgb, 1 - alpha, overlay, alpha, 0)
    return lung, blended

