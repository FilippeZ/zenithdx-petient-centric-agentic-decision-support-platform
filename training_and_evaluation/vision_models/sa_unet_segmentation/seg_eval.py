import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, Model
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score

# File paths
csv_path = "/storage/data2/up1084660/dataset/chest-x-ray-dataset-with-lung-segmentation-1.0.0/CXLSeg-segmented.csv"
raw_base_path = "/storage/data2/up1084660/dataset/chest-x-ray-dataset-with-lung-segmentation-1.0.0/mimic-cxr-jpg/2.1.0"
mask_base_path = "/storage/data2/up1084660/dataset/chest-x-ray-dataset-with-lung-segmentation-1.0.0/"

# Έλεγχος αν το CSV υπάρχει
if not os.path.exists(csv_path):
    sys.exit(f"Error: CSV file not found at {csv_path}")

# Φόρτωση των πρώτων 20.000 γραμμών του CSV
df = pd.read_csv(csv_path, skiprows=range(1, 50001), nrows=6250)

# Συνάρτηση για να φτιάχνει paths εικόνων
def get_image_paths(row):
    raw_image = os.path.join(raw_base_path, row["DicomPath"])
    mask_image = os.path.join(mask_base_path, row["DicomPath"].replace(".jpg", "-mask.jpg"))
    return raw_image, mask_image

df["RawImage"], df["MaskImage"] = zip(*df.apply(get_image_paths, axis=1))


# Προσθέτουμε μια στήλη που δείχνει αν το αρχείο εικόνας υπάρχει
df["exists"] = df["RawImage"].apply(os.path.exists)

# Φιλτράρουμε το DataFrame για να δούμε μόνο τις γραμμές με εικόνες που υπάρχουν
existing_images_df = df[df["exists"] == True]

# Εμφανίζουμε τις πρώτες γραμμές για έλεγχο
print(existing_images_df.head())

# Εμφανίζουμε τον συνολικό αριθμό εικόνων που υπάρχουν
print(f"Αριθμός εικόνων που υπάρχουν: {existing_images_df.shape[0]}")

# Συνάρτηση φόρτωσης και κανονικοποίησης εικόνας
def load_image(image_path):
    img = load_img(image_path, color_mode="grayscale", target_size=(256, 256))
    return img_to_array(img) / 255.0  # [0,1]

# Δημιουργία του dataset αξιολόγησης
def create_dataset(df, batch_size=16):
    def generator():
        for _, row in df.iterrows():
            raw = load_image(row["RawImage"])
            mask = load_image(row["MaskImage"])
            yield raw, mask

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32)
        )
    )
    return dataset.batch(batch_size)

dataset = create_dataset(df)

# Ορισμός του SA-UNet (χωρίς training section)
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.concat = layers.Concatenate(axis=-1)
        self.conv = layers.Conv2D(1, kernel_size=self.kernel_size, padding="same", activation="sigmoid")
        self.multiply = layers.Multiply()

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = self.concat([avg_pool, max_pool])
        attention = self.conv(concat)
        return self.multiply([inputs, attention])

def spatial_attention_block(x):
    return SpatialAttention()(x)

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    return x

def encoder_block(x, filters):
    s = conv_block(x, filters)
    p = layers.MaxPooling2D((2, 2))(s)
    return s, p

def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(x)
    skip = spatial_attention_block(skip)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

def build_sa_unet(input_shape=(256, 256, 1)):
    inputs = layers.Input(input_shape)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    bottleneck = conv_block(p4, 1024)
    d1 = decoder_block(bottleneck, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d4)
    return Model(inputs, outputs, name='SA-UNet')

# Δημιουργία μοντέλου και φόρτωση εκπαιδευμένων βαρών
unet_model = build_sa_unet(input_shape=(256, 256, 1))
unet_model.load_weights('/storage/data2/up1084660/dataset/chest-x-ray-dataset-with-lung-segmentation-1.0.0/unet_model.keras')

# ---- ΑΞΙΟΛΟΓΗΣΗ ----
y_true_seg = []
y_pred_seg = []

for X_batch, Y_true_batch in dataset:
    Y_pred_batch = unet_model.predict(X_batch)
    Y_pred_batch = (Y_pred_batch > 0.5).astype(np.uint8)  # Convert to binary
    y_true_np = Y_true_batch.numpy().flatten()
    y_pred_np = Y_pred_batch.flatten()
    y_true_seg.extend(y_true_np)
    y_pred_seg.extend(y_pred_np)

# Βεβαιώσου ότι είναι binary
y_true_seg = np.array(y_true_seg)
y_true_seg = (y_true_seg > 0.5).astype(np.uint8)
y_pred_seg = np.array(y_pred_seg)
y_pred_seg = (y_pred_seg > 0.5).astype(np.uint8)

# Υπολογισμός μετρικών
iou = jaccard_score(y_true_seg, y_pred_seg, average='binary')
dice = f1_score(y_true_seg, y_pred_seg, average='binary')
precision = precision_score(y_true_seg, y_pred_seg, average='binary')
recall = recall_score(y_true_seg, y_pred_seg, average='binary')

print("\nSegmentation Model Metrics:")
print(f"IoU (Jaccard Index): {iou:.4f}")
print(f"Dice Score (F1): {dice:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
