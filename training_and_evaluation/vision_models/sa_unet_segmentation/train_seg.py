import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import random
import torch.nn.functional as F
import multiprocessing

# ================= Multiprocessing spawn (safe for CUDA/fork) =================
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

# ================= Utilities =================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

def setup_gpu():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("No GPU available, using CPU")
        return False

# ================= Augmentation =================
def get_train_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# ================= Data Handling =================
def load_image_from_path(path, target_size=(224, 224)):
    if not isinstance(path, str) or not os.path.exists(path):
        return np.zeros((*target_size, 3), dtype=np.uint8)
    try:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != target_size:
            img = cv2.resize(img, target_size)
        return img
    except Exception as e:
        print(f"Error loading image: {path} | {e}")
        return np.zeros((*target_size, 3), dtype=np.uint8)

class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, label_cols, transform=None, sample_size=None, is_test=False):
        if sample_size:
            self.data = pd.read_csv(csv_path, nrows=sample_size)
        else:
            self.data = pd.read_csv(csv_path)
        self.label_cols = label_cols
        self.transform = transform
        self.is_test = is_test
        if self.is_test and 'is_augmented' in self.data.columns:
            self.data = self.data[self.data['is_augmented'] == 0].reset_index(drop=True)
        orig_len = len(self.data)
        self.data = self.data.dropna(subset=['abs_path'] + self.label_cols)
        for c in self.label_cols:
            self.data = self.data[self.data[c].astype(str).str.strip().isin(['0', '1', '0.0', '1.0'])]
        new_len = len(self.data)
        if orig_len != new_len:
            print(f"WARNING: Dropped {orig_len-new_len} rows due to label/path errors")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label_values = [float(str(row[col]).strip()) for col in self.label_cols]
        labels = torch.tensor(label_values, dtype=torch.float32)
        image = load_image_from_path(row['abs_path'])
        if self.transform:
            image = self.transform(image)
        return image, labels, idx

# ================== Class-balanced Sampler ==================
def get_class_balanced_sampler(dataset, label_cols):
    targets = dataset.data[label_cols].astype(float).values
    sample_weights = (targets / (targets.sum(axis=0) + 1e-8)).sum(axis=1)
    sample_weights = np.where(sample_weights > 0, 1.0 / sample_weights, 1.0)
    sample_weights = sample_weights / np.sum(sample_weights)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

# ================== Focal Loss ==================
class FocalLoss(nn.Module):
    def __init__(self, pos_weight=None, gamma=2, reduction='mean'):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_factor = (1 - p_t) ** self.gamma
        loss = focal_factor * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ================== Model and Training ==================
def create_model(num_classes, pretrained=True, freeze_backbone=False):
    model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    return model

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        self.dataset = dataloader.dataset
        self.batch_size = dataloader.batch_size
    def __iter__(self):
        for batch in self.dataloader:
            yield to_device(batch, self.device)
    def __len__(self):
        return len(self.dataloader)

# =============== MixUp ===============
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# =============== Metric Computation ===============
def compute_metrics(preds, labels):
    preds = 1 / (1 + np.exp(-preds))
    preds = (preds > 0.5).astype(int)
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

def train_one_epoch(model, dataloader, criterion, optimizer, device, use_mixup=True):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_logits = []
    total_samples = 0
    for images, labels, _ in tqdm(dataloader, desc="Training"):
        batch_size = images.size(0)
        total_samples += batch_size
        optimizer.zero_grad()
        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels)
            images, labels_a, labels_b = images.to(device), labels_a.to(device), labels_b.to(device)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            all_logits.append(outputs.detach().cpu().numpy())
            all_labels.append(labels_a.cpu().detach().numpy())
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            all_logits.append(outputs.detach().cpu().numpy())
            all_labels.append(labels.cpu().detach().numpy())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_size
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_logits, all_labels)
    epoch_loss = running_loss / total_samples
    return epoch_loss, metrics

def evaluate(model, dataloader, criterion, device, label_cols, best_thresholds=None):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_logits = []
    total_samples = 0
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Evaluating"):
            batch_size = images.size(0)
            total_samples += batch_size
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * batch_size
            all_logits.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_logits, all_labels)
    per_class = {}
    probs = 1 / (1 + np.exp(-all_logits))
    preds = (probs > (best_thresholds if best_thresholds is not None else 0.5)).astype(int)
    for i, label in enumerate(label_cols):
        per_class[label] = {
            "accuracy": accuracy_score(all_labels[:, i], preds[:, i]),
            "precision": precision_score(all_labels[:, i], preds[:, i], zero_division=0),
            "recall": recall_score(all_labels[:, i], preds[:, i], zero_division=0),
            "f1": f1_score(all_labels[:, i], preds[:, i], zero_division=0)
        }
    epoch_loss = running_loss / total_samples
    return epoch_loss, metrics, per_class, all_logits, all_labels

def find_best_thresholds(val_logits, val_labels, label_cols):
    best_thresholds = []
    probs = 1 / (1 + np.exp(-val_logits))
    for i, label in enumerate(label_cols):
        truths = val_labels[:, i]
        best_f1, best_th = 0, 0.5
        for th in np.linspace(0.1, 0.9, 41):
            preds = (probs[:, i] > th).astype(int)
            f1 = f1_score(truths, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_th = f1, th
        best_thresholds.append(best_th)
        print(f"{label}: best threshold = {best_th:.2f}, F1 = {best_f1:.3f}")
    return np.array(best_thresholds)

# ============= Plotting & Saving =============
def plot_confusion_matrices(cms, label_cols, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i, label in enumerate(label_cols):
        plt.figure(figsize=(8, 6))
        cm = cms[label]
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        sns.heatmap(
            cm_norm,
            annot=cm,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive']
        )
        plt.title(f'Confusion Matrix: {label}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'confusion_matrix_{label}.png'))
        plt.close()

def plot_metrics_history(train_losses, val_losses, train_metrics, val_metrics, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_history.png'))
    plt.close()
    plt.figure(figsize=(10, 6))
    plt.plot([m['accuracy'] for m in train_metrics], label='Train Accuracy')
    plt.plot([m['accuracy'] for m in val_metrics], label='Validation Accuracy')
    plt.title('Accuracy over Training')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy_history.png'))
    plt.close()
    plt.figure(figsize=(10, 6))
    plt.plot([m['f1_score'] for m in train_metrics], label='Train F1 Score')
    plt.plot([m['f1_score'] for m in val_metrics], label='Validation F1 Score')
    plt.title('F1 Score over Training')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'f1_history.png'))
    plt.close()

def save_metrics_to_json(metrics, filename):
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    serializable_metrics = {k: v for k, v in metrics.items() if k in ['loss','accuracy','precision','recall','f1_score','class_metrics']}
    serializable_metrics = convert_to_serializable(serializable_metrics)
    with open(filename, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    print(f"Metrics saved to {filename}")

def get_pos_weights(label_cols, train_dataset, device):
    pos_weights = []
    for label in label_cols:
        pos_count = train_dataset.data[label].astype(float).sum()
        neg_count = len(train_dataset.data) - pos_count
        weight = neg_count / pos_count if pos_count > 0 else 1.0
        pos_weights.append(weight)
    pos_weights = torch.tensor(pos_weights, dtype=torch.float32).to(device)
    print("Class weights for loss function:")
    for label, weight in zip(label_cols, pos_weights.cpu().numpy()):
        print(f"{label}: {weight:.2f}")
    return pos_weights

def verify_dataset(dataset, num_samples=5):
    print(f"\nVerifying dataset with {len(dataset)} samples")
    if hasattr(dataset, 'data') and 'abs_path' in dataset.data.columns:
        num_empty = dataset.data['abs_path'].isna().sum()
        print(f"Number of empty image paths: {num_empty} out of {len(dataset)}")
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        for i, idx in enumerate(indices):
            img, labels, _ = dataset[idx]
            if isinstance(img, torch.Tensor):
                if torch.all(img == 0) or torch.isnan(img).any():
                    print(f"Sample {i+1}: Empty or invalid image tensor")
                else:
                    print(f"Sample {i+1}: Valid image tensor with shape {img.shape}")
            else:
                print(f"Sample {i+1}: Image is not a tensor, type: {type(img)}")
            pos_labels = [dataset.label_cols[j] for j in range(len(labels)) if labels[j] == 1]
            print(f"  Positive labels: {pos_labels}")

# ================== Main Training Loop ==================
def main():
    set_seed(42)
    has_gpu = setup_gpu()
    device = torch.device("cuda" if has_gpu else "cpu")
    print(f"Using device: {device}")

    base_dir = "/storage/data2/up1084660/dataset/chest-x-ray-dataset-with-lung-segmentation-1.0.0"
    data_dir = os.path.join(base_dir, "processed_seg")
    output_dir = os.path.join(base_dir, "model_outputs", "resnet50_aug_balance")
    os.makedirs(output_dir, exist_ok=True)
    sample_size = None

    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")

    df_sample = pd.read_csv(train_csv, nrows=100)
    label_cols = [col for col in df_sample.columns if col not in ['abs_path', 'original_idx', 'is_augmented', 'aug_idx', 'image', 'image_path']]
    print(f"Found {len(label_cols)} label columns: {label_cols}")

    train_dataset = ChestXrayDataset(train_csv, label_cols, transform=get_train_transforms(), sample_size=sample_size, is_test=False)
    val_dataset = ChestXrayDataset(test_csv, label_cols, transform=get_val_transforms(), sample_size=sample_size, is_test=True)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    print("\nVerifying training dataset:")
    verify_dataset(train_dataset)
    print("\nVerifying validation dataset:")
    verify_dataset(val_dataset)

    batch_size = 32
    pos_weights = get_pos_weights(label_cols, train_dataset, device)
    sampler = get_class_balanced_sampler(train_dataset, label_cols)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    val_loader = DeviceDataLoader(val_loader, device)

    num_classes = len(label_cols)
    model = create_model(num_classes, pretrained=True, freeze_backbone=False)
    model = model.to(device)
    print(f"Created ResNet-50 model with {num_classes} output classes")

    if has_gpu:
        print("\nGPU Memory Usage Before Training:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    criterion = FocalLoss(pos_weight=pos_weights, gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    num_epochs = 15

    train_losses, val_losses = [], []
    train_metrics_history, val_metrics_history = [], []

    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    cm_dir = os.path.join(output_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)

    best_val_f1 = 0.0
    best_epoch = -1
    best_model_path = os.path.join(models_dir, 'best_model.pth')
    best_thresholds = None

    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_mixup=True
        )
        train_losses.append(train_loss)
        train_metrics_history.append(train_metrics)

        val_loss, val_metrics, val_per_class, val_logits, val_labels = evaluate(
            model, val_loader, criterion, device, label_cols
        )
        best_thresholds = find_best_thresholds(val_logits, val_labels, label_cols)
        val_losses.append(val_loss)
        val_metrics_history.append(val_metrics)

        scheduler.step(val_loss)
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

        if has_gpu:
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated")

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            print(f"Saving new best model with F1: {val_metrics['f1']:.4f}")
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': val_metrics['f1'],
                    'val_loss': val_loss,
                    'label_cols': label_cols,
                    'best_thresholds': best_thresholds.tolist(),
                }, best_model_path)
                print(f"Model successfully saved to {best_model_path}")
            except Exception as e:
                print(f"Error saving model: {e}")

    print(f"\nTraining completed. Best model from epoch {best_epoch+1} with F1: {best_val_f1:.4f}")

    try:
        plot_metrics_history(
            train_losses,
            val_losses,
            train_metrics_history,
            val_metrics_history,
            plots_dir
        )
        print(f"Training plots saved to {plots_dir}")
    except Exception as e:
        print(f"Error creating plots: {e}")

    final_model = model
    if best_epoch >= 0:
        print(f"Attempting to load best model from epoch {best_epoch+1}...")
        try:
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                best_thresholds = np.array(checkpoint.get('best_thresholds', [0.5]*len(label_cols)))
                final_model = model
                print(f"Successfully loaded best model from epoch {best_epoch+1}")
            else:
                print(f"Best model file not found at {best_model_path}. Using current model for evaluation.")
        except Exception as e:
            print(f"Error loading best model: {e}. Using current model for evaluation.")
    else:
        print("No best model was saved during training. Using current model for evaluation.")

    print("\nPerforming final evaluation on test set...")
    test_loss, test_metrics, test_per_class, test_logits, test_labels = evaluate(
        final_model, val_loader, criterion, device, label_cols, best_thresholds=best_thresholds
    )

    print("\nFinal Test Metrics:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")

    print("\nPer-Class Metrics:")
    for label, metrics in test_per_class.items():
        print(f"\n{label}:")
        for k, v in metrics.items():
            print(f"  {k.capitalize()}: {v:.4f}")

    try:
        plot_confusion_matrices(
            {label: confusion_matrix(test_labels[:, i], (1 / (1 + np.exp(-test_logits[:, i])) > best_thresholds[i]).astype(int)) for i, label in enumerate(label_cols)},
            label_cols,
            cm_dir
        )
        print(f"Confusion matrices saved to {cm_dir}")
    except Exception as e:
        print(f"Error creating confusion matrices: {e}")

    try:
        metrics_path = os.path.join(output_dir, 'test_metrics.json')
        save_metrics_to_json({
            'loss': test_loss,
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1_score': test_metrics['f1'],
            'class_metrics': test_per_class
        }, metrics_path)
        print(f"Test metrics saved to {metrics_path}")
    except Exception as e:
        print(f"Error saving metrics: {e}")

    print("\nAll results saved to:", output_dir)

    final_model_path = os.path.join(models_dir, 'final_model.pth')
    try:
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'label_cols': label_cols,
            'best_thresholds': best_thresholds.tolist() if best_thresholds is not None else [0.5]*len(label_cols)
        }, final_model_path)
        print(f"Final model saved to {final_model_path}")
    except Exception as e:
        print(f"Error saving final model: {e}")

if __name__ == "__main__":
    main()
