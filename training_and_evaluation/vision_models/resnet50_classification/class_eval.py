import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, Model, ops
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.saving import register_keras_serializable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# -------------------------------
# Section 1: Data Loading & Preprocessing for Evaluation
# -------------------------------
csv_path = "/storage/data2/up1084660/dataset/chest-x-ray-dataset-with-lung-segmentation-1.0.0/CXLSeg-train-updated.csv"
base_path = "/storage/data2/up1084660/dataset/chest-x-ray-dataset-with-lung-segmentation-1.0.0/"

# Load only 6,250 evaluation samples (skip the first 50,000 rows; row 0 is the header)
df = pd.read_csv(csv_path, skiprows=range(1, 50001), nrows=6250)

def get_raw_image_path(row):
    return os.path.join(base_path, row["DicomPath"])

# Add a column with the raw image path
df["RawImage"] = df.apply(get_raw_image_path, axis=1)

# -------------------------------
# Section 2: Dataset Creation for Evaluation
# -------------------------------
def load_image(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))
        return img_to_array(img) / 255.0
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def create_dataset(df, batch_size=16):
    def generator():
        for _, row in df.iterrows():
            img = load_image(row["RawImage"])
            if img is None:
                continue
            # Choose multi-label classification targets
            label = row[['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
                         'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
                         'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 
                         'Pneumonia', 'Pneumothorax', 'Support Devices']].values.astype(float)
            label = np.nan_to_num(label, nan=0.0)
            yield img, label

    dataset = tf.data.Dataset.from_generator(generator,
                                             output_signature=(
                                                 tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(14,), dtype=tf.float32)
                                             ))
    return dataset.batch(batch_size).shuffle(100)

eval_dataset = create_dataset(df, batch_size=16)
# Calculate the number of evaluation steps (approximate)
num_samples = len(df)
eval_steps = num_samples // 16

# -------------------------------
# Section 3: Build Vision Transformer (ViT) Model for Classification
# -------------------------------
input_shape = (224, 224, 3)
image_size = 72    # Resize images for patch extraction
patch_size = 6     # Size of each patch
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [projection_dim * 2, projection_dim]  # Transformer MLP units
transformer_layers = 8
mlp_head_units = [2048, 1024]

@register_keras_serializable()
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

@register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = ops.expand_dims(ops.arange(start=0, stop=self.num_patches, step=1), axis=0)
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def build_vit_model(num_classes):
    inputs = keras.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    patches = Patches(patch_size)(augmented)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes, activation='sigmoid')(features)
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

vit_model = build_vit_model(num_classes=14)
# Load pre-trained weights if available (ensure compatibility)
vit_model.load_weights('vit_model.weights.h5')
vit_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# Section 4: Evaluate the ViT Model on 6,250 Samples
# -------------------------------
y_true_cls = []
y_pred_cls = []   # Thresholded (binary) predictions for accuracy, etc.
y_pred_probs = [] # Continuous probability outputs for ROC AUC

# Iterate through the entire evaluation dataset (all batches)
for X_batch, Y_true_batch in eval_dataset.take(eval_steps):
    probs_batch = vit_model.predict(X_batch)
    preds_batch = (probs_batch > 0.5).astype(int)
    
    y_true_cls.append(Y_true_batch.numpy())
    y_pred_cls.append(preds_batch)
    y_pred_probs.append(probs_batch)
    
    print("Predicted shape:", preds_batch.shape)
    print("True label shape:", Y_true_batch.shape)

y_true_cls = np.vstack(y_true_cls)
y_pred_cls = np.vstack(y_pred_cls)
y_pred_probs = np.vstack(y_pred_probs)

print("Final shapes:")
print("y_true_cls:", y_true_cls.shape)  # Expected shape: (~6250, 14)
print("y_pred_cls:", y_pred_cls.shape)  # Expected shape: (~6250, 14)

# Compute standard classification metrics on binary predictions
accuracy = accuracy_score(y_true_cls, y_pred_cls)
precision = precision_score(y_true_cls, y_pred_cls, average="weighted", zero_division=0)
recall = recall_score(y_true_cls, y_pred_cls, average="weighted", zero_division=0)
f1 = f1_score(y_true_cls, y_pred_cls, average="weighted", zero_division=0)

print("\nClassification Model Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Compute ROC AUC metrics using the continuous probability outputs
# Note: ROC AUC can be computed in a multi-label setting by specifying an average.
try:
    auc_macro = roc_auc_score(y_true_cls, y_pred_probs, average='macro')
    auc_micro = roc_auc_score(y_true_cls, y_pred_probs, average='micro')
    print(f"AUC Macro: {auc_macro:.4f}")
    print(f"AUC Micro: {auc_micro:.4f}")
except ValueError as e:
    print("Error computing ROC AUC:", e)

# Compute per-class AUC scores
auc_per_class = {}
for i in range(y_true_cls.shape[1]):
    try:
        auc = roc_auc_score(y_true_cls[:, i], y_pred_probs[:, i])
        auc_per_class[f"Class_{i}"] = auc
    except ValueError as e:
        auc_per_class[f"Class_{i}"] = None
print("\nPer-class AUC scores:")
for cls, auc in auc_per_class.items():
    print(f"{cls}: {auc:.4f}" if auc is not None else f"{cls}: N/A")

# Optionally, print a detailed classification report per class
print("\nDetailed Classification Report:")
print(classification_report(y_true_cls, y_pred_cls, zero_division=0))
