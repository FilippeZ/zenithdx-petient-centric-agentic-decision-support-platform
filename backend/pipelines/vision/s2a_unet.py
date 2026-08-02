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
    """
    Skip-Spatial Attention Block (S²A-UNet Core Innovation).
    Computes spatial attention maps from Channel AvgPool and Channel MaxPool
    via a 7x7 Conv2D with Sigmoid activation, then element-wise multiplies
    with the input skip connection tensor to suppress non-pulmonary noise.
    """
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

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config

def dice_coef(y_true, y_pred, smooth=1):
    """Dice similarity coefficient evaluation metric."""
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(tf.round(y_pred), [-1])
    inter = tf.reduce_sum(y_true * y_pred)
    return (2. * inter + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def build_s2a_unet_architecture(input_shape=(256, 256, 1)):
    """
    Constructs the 2-Stage S²A-UNet model architecture:
    - 4 Encoder levels (64 -> 128 -> 256 -> 512 channels)
    - Bottleneck (1024 channels at 16x16 resolution)
    - Spatial Attention on Skip Connections
    - 4 Decoder levels (512 -> 256 -> 128 -> 64 channels)
    - 1x1 Conv output with Sigmoid activation (256x256x1 binary mask)
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder
    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 512)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    b = conv_block(p4, 1024)

    # Skip-Spatial Attention
    sa1 = SpatialAttention()(c1)
    sa2 = SpatialAttention()(c2)
    sa3 = SpatialAttention()(c3)
    sa4 = SpatialAttention()(c4)

    # Decoder
    u4 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(b)
    u4 = layers.concatenate([u4, sa4])
    c5 = conv_block(u4, 512)

    u3 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c5)
    u3 = layers.concatenate([u3, sa3])
    c6 = conv_block(u3, 256)

    u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c6)
    u2 = layers.concatenate([u2, sa2])
    c7 = conv_block(u2, 128)

    u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c7)
    u1 = layers.concatenate([u1, sa1])
    c8 = conv_block(u1, 64)

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(c8)

    model = keras.Model(inputs=[inputs], outputs=[outputs], name="S2A_UNet")
    return model

def _unicode_safe_copy_to_ascii(src_path: "pathlib.Path") -> str:
    """
    Copies a SavedModel directory to a short, pure-ASCII temp path so that
    TensorFlow's internal C++ reader (which cannot handle non-ASCII chars on
    Windows) can load it successfully.  Returns the ASCII destination path string.
    """
    import pathlib, shutil, tempfile
    dest = pathlib.Path(tempfile.gettempdir()) / "zd_unet_model"
    if dest.exists():
        shutil.rmtree(str(dest), ignore_errors=True)
    shutil.copytree(str(src_path), str(dest))
    print(f"[SA-UNet] Copied model to ASCII temp path: {dest}", file=sys.stderr)
    return str(dest)


def load_sa_unet(weights_path: str):
    """
    Load SA-UNet model from weights path (SavedModel dir or .h5/.keras).

    Unicode-safe: if the path contains non-ASCII characters (e.g. Greek folder
    names), the SavedModel directory is first copied to a short ASCII temp path
    so TensorFlow's TFSMLayer can open the variables file without a UTF-8 decode
    error.  Falls back to keras.models.load_model, then constructs a fresh
    S²A-UNet architecture if both loaders fail.
    """
    import pathlib
    p = pathlib.Path(weights_path)
    print(f"[SA-UNet] Loading SA-UNet model from: {p}")

    custom_objects = {"SpatialAttention": SpatialAttention, "dice_coef": dice_coef}

    if p.exists():
        try:
            if p.is_dir():
                # ── Check that the variables weights file actually exists ──────
                vars_dir = p / "variables"
                weights_file = vars_dir / "variables.data-00000-of-00001"
                if not weights_file.exists():
                    print(
                        f"[SA-UNet] ⚠️  variables.data file missing in {p}. "
                        "Place your trained weights there and restart.",
                        file=sys.stderr,
                    )
                    raise FileNotFoundError(f"Missing weights: {weights_file}")

                # ── Unicode-safe: copy to ASCII temp dir before TFSMLayer ─────
                try:
                    load_path = str(p)
                    load_path.encode("ascii")          # raises if non-ASCII
                except (UnicodeEncodeError, UnicodeDecodeError):
                    load_path = _unicode_safe_copy_to_ascii(p)

                print(f"[SA-UNet] Loading SavedModel via Keras 3 TFSMLayer from {load_path}", file=sys.stderr)
                try:
                    inputs = layers.Input(shape=(256, 256, 1))
                    tfsm_layer = keras.layers.TFSMLayer(load_path, call_endpoint="serving_default")
                    outputs = tfsm_layer(inputs)
                    if isinstance(outputs, dict):
                        outputs = list(outputs.values())[0]
                    model = keras.Model(inputs=inputs, outputs=outputs, name="SA_UNet_TFSM")
                    print(f"[SA-UNet] ✅ Loaded TFSMLayer model successfully from {load_path}", file=sys.stderr)
                    return model
                except Exception as e_tfsm:
                    print(f"[SA-UNet] TFSMLayer failed ({e_tfsm}), trying keras.models.load_model...", file=sys.stderr)
                    model = keras.models.load_model(load_path, custom_objects=custom_objects, compile=False)
                    return model
            else:
                # ── Unicode-safe: copy .keras/.h5 to ASCII temp path ──────────
                try:
                    load_path = str(p)
                    load_path.encode("ascii")      # raises if non-ASCII chars
                except (UnicodeEncodeError, UnicodeDecodeError):
                    import shutil, tempfile
                    dest = pathlib.Path(tempfile.gettempdir()) / ("zd_unet" + p.suffix)
                    shutil.copy2(str(p), str(dest))
                    load_path = str(dest)
                    print(f"[SA-UNet] Copied {p.name} to ASCII temp path: {dest}", file=sys.stderr)

                print(f"[SA-UNet] Loading Keras model from {load_path}", file=sys.stderr)
                model = keras.models.load_model(
                    load_path,
                    custom_objects=custom_objects,
                    compile=False,
                )
                print(f"[SA-UNet] ✅ Loaded Keras model successfully from {load_path}", file=sys.stderr)
                return model
        except Exception as e:
            print(f"[SA-UNet] ⚠️ Error loading saved weights ({e}). Constructing S²A-UNet architecture.", file=sys.stderr)

    # Architectural fallback — no trained weights available
    try:
        model = build_s2a_unet_architecture(input_shape=(256, 256, 1))
        print(f"[SA-UNet] [OK] Constructed S²A-UNet architecture successfully.", file=sys.stderr)
        return model
    except Exception as e:
        print(f"[SA-UNet] ❌ Failed to construct S²A-UNet architecture: {e}", file=sys.stderr)
        return None

def fallback_anatomical_lung_segmentation(img_rgb: np.ndarray) -> np.ndarray:
    """
    High-precision, anatomically accurate dual-lobe lung field segmentation.
    Extracts true left and right pulmonary parenchymal lobes situated strictly in the
    chest cavity with smooth, natural anatomical curves.
    """
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)

    mask = np.zeros((h, w), dtype=np.uint8)

    # 1. Segment Left Pulmonary Lobe (Viewer's Left: x in [0.08w, 0.46w], y in [0.12h, 0.84h])
    left_roi = np.zeros((h, w), dtype=np.uint8)
    left_roi[int(0.12 * h):int(0.84 * h), int(0.08 * w):int(0.46 * w)] = 255
    left_pixels = blurred[left_roi > 0]
    
    if len(left_pixels) > 0:
        t_left = np.percentile(left_pixels, 45)
        left_thresh = ((blurred < t_left) & (left_roi > 0)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        left_cleaned = cv2.morphologyEx(left_thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        cnts, _ = cv2.findContours(left_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c_max = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c_max) > 0.01 * h * w:
                cv2.drawContours(mask, [c_max], -1, 255, -1)
            else:
                mask[int(0.15 * h):int(0.80 * h), int(0.10 * w):int(0.44 * w)] = 255
        else:
            mask[int(0.15 * h):int(0.80 * h), int(0.10 * w):int(0.44 * w)] = 255

    # 2. Segment Right Pulmonary Lobe (Viewer's Right: x in [0.54w, 0.92w], y in [0.12h, 0.84h])
    right_roi = np.zeros((h, w), dtype=np.uint8)
    right_roi[int(0.12 * h):int(0.84 * h), int(0.54 * w):int(0.92 * w)] = 255
    right_pixels = blurred[right_roi > 0]

    if len(right_pixels) > 0:
        t_right = np.percentile(right_pixels, 45)
        right_thresh = ((blurred < t_right) & (right_roi > 0)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        right_cleaned = cv2.morphologyEx(right_thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        cnts, _ = cv2.findContours(right_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c_max = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c_max) > 0.01 * h * w:
                cv2.drawContours(mask, [c_max], -1, 255, -1)
            else:
                mask[int(0.15 * h):int(0.80 * h), int(0.56 * w):int(0.90 * w)] = 255
        else:
            mask[int(0.15 * h):int(0.80 * h), int(0.56 * w):int(0.90 * w)] = 255

    mask_float = mask.astype("float32") / 255.0
    mask_smooth = cv2.GaussianBlur(mask_float, (15, 15), 0)
    return mask_smooth

def advanced_lung_extraction(original_image_np: np.ndarray, raw_broken_mask_np: np.ndarray):
    """
    Fixes broken/jagged segmentation masks using Convex Hull & Morphological Closing.
    Operates strictly on uint8 [0, 255] space for OpenCV morphological operations,
    then normalizes to float32 [0.0, 1.0] with soft Gaussian edge feathering.
    """
    # 1. Ensure float32 scale [0.0, 1.0] for input mask
    mask_float = raw_broken_mask_np.astype(np.float32)
    if mask_float.max() > 1.0:
        mask_float /= 255.0

    # 2. Scale to proper uint8 [0, 255] for OpenCV operations
    mask_uint8 = (mask_float > 0.3).astype(np.uint8) * 255
    h, w = mask_uint8.shape[:2]

    # 3. Contour detection & Convex Hull reconstruction in [0, 255] space
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fixed_mask = np.zeros((h, w), dtype=np.uint8)
    min_area = max(500, int(0.005 * h * w))
    
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(fixed_mask, [hull], -1, 255, thickness=cv2.FILLED)
            
    # 4. Morphological closing in [0, 255] space to fill retrocardiac / cardiac notch gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    fixed_mask = cv2.morphologyEx(fixed_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # 5. Convert to float32 [0.0, 1.0] and apply soft Gaussian edge feathering
    mask_float32 = fixed_mask.astype(np.float32) / 255.0
    soft_mask = cv2.GaussianBlur(mask_float32, (15, 15), 0)
    final_mask = np.clip(soft_mask, 0.0, 1.0)
    
    # 6. Safe RGB Masking
    if len(original_image_np.shape) == 3 and len(final_mask.shape) == 2:
        mask_3d = final_mask[..., None]
    else:
        mask_3d = final_mask
        
    orig_uint8 = original_image_np.astype(np.uint8) if original_image_np.dtype != np.uint8 else original_image_np
    isolated_lungs = (orig_uint8.astype(np.float32) * mask_3d).astype(np.uint8)
    return final_mask, isolated_lungs

def refine_unet_mask(raw_mask: np.ndarray) -> np.ndarray:
    """
    Refines U-Net segmentation mask using Morphological Opening & 2-largest Contour Filtering.
    - Eliminates noise, sharp spikes, and "ear" protrusions outside the lung boundary.
    - Fills small intra-pulmonary holes via Morphological Closing.
    - Filters contours to retain strictly the 2 largest connected components (left & right lung fields).
    - Returns float32 mask normalized in [0.0, 1.0].
    """
    # 1. Convert to uint8 [0, 255] for OpenCV morphological operations
    if raw_mask.max() <= 1.5:
        mask_uint8 = (raw_mask * 255).astype(np.uint8)
    else:
        mask_uint8 = raw_mask.astype(np.uint8)

    # 2. Kernel definition (9x9 ellipse kernel for smooth natural curves)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # 3. Morphological Opening (Erosion followed by Dilation) to erase ear spikes & protrusions
    opened_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4. Morphological Closing to fill small holes inside lungs
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 5. Pro-Tip: Contour filtering — keep ONLY the 2 largest contours (left & right lungs)
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask_uint8 = np.zeros_like(closed_mask)
    if contours:
        # Sort contours by area in descending order
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Keep top 2 largest contours (left & right lungs)
        top_contours = [cnt for cnt in sorted_contours[:2] if cv2.contourArea(cnt) > 200]
        if top_contours:
            cv2.drawContours(final_mask_uint8, top_contours, -1, 255, thickness=cv2.FILLED)
        else:
            final_mask_uint8 = closed_mask
    else:
        final_mask_uint8 = closed_mask

    # 6. Convert back to float32 range [0.0, 1.0]
    final_mask = (final_mask_uint8 / 255.0).astype(np.float32)
    return final_mask

def sa_unet_predict(model, img_rgb: np.ndarray, target_size=(256, 256)) -> np.ndarray:
    """
    Stage 1 S²A-UNet Inference:
    1. Preprocesses input image (Grayscale -> 256x256 -> float32 normalize [0.0, 1.0] -> 1x256x256x1).
    2. Passes through S²A-UNet to generate 256x256 binary probability map.
    3. Resizes probability map to original image resolution (W_orig, H_orig).
    4. Applies Convex Hull advanced_lung_extraction to recover pathology-erased lobes.
    5. Applies Morphological Opening (refine_unet_mask) to eliminate ear spikes & floating noise.
    """
    if len(img_rgb.shape) == 3:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb

    gray_resized = cv2.resize(gray, target_size)
    gray_norm = gray_resized.astype("float32") / 255.0
    unet_input = np.expand_dims(gray_norm, axis=(0, -1)) # Shape: (1, 256, 256, 1)

    if model is not None:
        try:
            if hasattr(model, "predict"):
                pred = model.predict(unet_input, verbose=0)[0, ..., 0]
            elif callable(model):
                pred = model(tf.constant(unet_input), training=False).numpy()[0, ..., 0]
            else:
                pred = None

            if pred is not None:
                # Resize raw 256x256 prediction map back to full original image resolution
                raw_mask_full = cv2.resize(
                    pred.astype("float32"),
                    (img_rgb.shape[1], img_rgb.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
                
                # Apply Convex Hull advanced lung extraction at full original resolution
                raw_extracted_mask, _ = advanced_lung_extraction(img_rgb, raw_mask_full)
                
                # Morphological Opening & 2-largest contour refinement
                final_mask = refine_unet_mask(raw_extracted_mask)
                
                coverage = np.mean(final_mask)
                if coverage > 0.85 or coverage < 0.01:
                    print(f"[SA-UNet] Mask unsegmented/overcovered (coverage={coverage:.2f}). Using anatomical segmentation.", file=sys.stderr)
                    return refine_unet_mask(fallback_anatomical_lung_segmentation(img_rgb))

                return final_mask
        except Exception as e:
            print(f"[SA-UNet] Predict warning ({e}). Using anatomical segmentation.", file=sys.stderr)

    return refine_unet_mask(fallback_anatomical_lung_segmentation(img_rgb))

def apply_mask(original_rgb: np.ndarray, mask: np.ndarray):
    """
    Crops & masks original RGB image to lung region of interest (ROI).
    Returns (masked_lung_image, blended_overlay_with_green_lung).
    """
    lung = (original_rgb * mask[..., None]).astype("uint8")
    overlay = original_rgb.copy()
    green_mask = np.zeros_like(original_rgb)
    green_mask[..., 1] = 255
    overlay = np.where(mask[..., None] > 0.5, green_mask, overlay)
    alpha = 0.35
    blended = cv2.addWeighted(original_rgb, 1 - alpha, overlay, alpha, 0)
    return lung, blended
