#!/usr/bin/env python3
"""
Ceramic Classification ML Model v2 - Enhanced Training
Multi-task CNN classifier with comprehensive cross-validation
Classifies: macro_period, decoration, AND vessel_type
With overfitting analysis and confusion matrices
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("    CERAMIC CLASSIFICATION ML MODEL v2 - ENHANCED TRAINING")
print("    Multi-task: Period + Decoration + Vessel Type")
print("=" * 70)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# CONFIGURATION - Enhanced Parameters
# ============================================
IMG_SIZE = 224
BATCH_SIZE = 32          # Increased from 16
EPOCHS = 50              # Increased from 25
N_FOLDS = 5
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4      # L2 regularization
EARLY_STOP_PATIENCE = 10 # Increased patience
MIN_SAMPLES_PER_CLASS = 5

DESKTOP_PATH = os.path.expanduser("~/Desktop")
MODEL_DIR = os.path.join(DESKTOP_PATH, "CeramicaML_Model_v2")
os.makedirs(MODEL_DIR, exist_ok=True)

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch: {torch.__version__} | Device: {device}")
print(f"Output: {MODEL_DIR}")

# ============================================
# LOAD AND PREPARE DATA
# ============================================
print("\n[1/7] Loading data from database...")
conn = sqlite3.connect('ceramica.db')
df = pd.read_sql_query("""
    SELECT id, collection, macro_period, decoration, vessel_type, image_path
    FROM items
    WHERE image_path IS NOT NULL AND image_path != ''
""", conn)
conn.close()

print(f"   Total items: {len(df)}")

# Clean data - fill missing values
df['macro_period'] = df['macro_period'].fillna('Unknown')
df['decoration'] = df['decoration'].fillna('plain')
df['vessel_type'] = df['vessel_type'].fillna('unknown')

# Filter classes with minimum samples
print("\n   Filtering classes with minimum samples...")

def filter_rare_classes(df, column, min_samples):
    counts = df[column].value_counts()
    valid_classes = counts[counts >= min_samples].index
    return df[df[column].isin(valid_classes)]

df_filtered = df.copy()
for col in ['macro_period', 'decoration', 'vessel_type']:
    before = len(df_filtered)
    df_filtered = filter_rare_classes(df_filtered, col, MIN_SAMPLES_PER_CLASS)
    after = len(df_filtered)
    if before != after:
        print(f"      {col}: removed {before - after} items with rare classes")

df = df_filtered
print(f"   Final dataset: {len(df)} items")

# Encode labels
le_period = LabelEncoder()
le_decoration = LabelEncoder()
le_vessel = LabelEncoder()

df['period_encoded'] = le_period.fit_transform(df['macro_period'])
df['decoration_encoded'] = le_decoration.fit_transform(df['decoration'])
df['vessel_encoded'] = le_vessel.fit_transform(df['vessel_type'])

n_period = len(le_period.classes_)
n_decoration = len(le_decoration.classes_)
n_vessel = len(le_vessel.classes_)

print(f"\n   Classes:")
print(f"      Period ({n_period}): {list(le_period.classes_)}")
print(f"      Decoration ({n_decoration}): {list(le_decoration.classes_)}")
print(f"      Vessel Type ({n_vessel}): {list(le_vessel.classes_)}")

# Class distribution
print(f"\n   Class Distribution:")
print(f"      Period: {dict(df['macro_period'].value_counts())}")
print(f"      Decoration: {dict(df['decoration'].value_counts())}")
print(f"      Vessel Type: {dict(df['vessel_type'].value_counts())}")

# ============================================
# DATASET CLASS
# ============================================
class CeramicDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        try:
            img = Image.open(row['image_path']).convert('RGB')
        except:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))

        if self.transform:
            img = self.transform(img)

        return (img,
                torch.tensor(row['period_encoded'], dtype=torch.long),
                torch.tensor(row['decoration_encoded'], dtype=torch.long),
                torch.tensor(row['vessel_encoded'], dtype=torch.long))

# ============================================
# DATA AUGMENTATION - Enhanced
# ============================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1)
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================
# MODEL - Multi-task with ResNet50
# ============================================
class CeramicClassifierV2(nn.Module):
    def __init__(self, n_period, n_decoration, n_vessel, dropout=0.4):
        super().__init__()

        # Use ResNet50 for better features
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        n_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Identity()

        # Shared layers with more capacity
        self.shared = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classification heads
        self.period_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(128, n_period)
        )

        self.decoration_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(128, n_decoration)
        )

        self.vessel_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(128, n_vessel)
        )

    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared(features)
        return (self.period_head(shared),
                self.decoration_head(shared),
                self.vessel_head(shared))

# ============================================
# TRAINING FUNCTIONS
# ============================================
def train_epoch(model, loader, criterion, optimizer, device, pbar_desc="Training"):
    model.train()
    total_loss = 0
    correct = {'period': 0, 'decoration': 0, 'vessel': 0}
    total = 0

    pbar = tqdm(loader, desc=pbar_desc, leave=False, ncols=100)
    for images, period_y, decoration_y, vessel_y in pbar:
        images = images.to(device)
        period_y = period_y.to(device)
        decoration_y = decoration_y.to(device)
        vessel_y = vessel_y.to(device)

        optimizer.zero_grad()
        period_out, decoration_out, vessel_out = model(images)

        loss = (criterion(period_out, period_y) +
                criterion(decoration_out, decoration_y) +
                criterion(vessel_out, vessel_y))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        correct['period'] += (period_out.argmax(1) == period_y).sum().item()
        correct['decoration'] += (decoration_out.argmax(1) == decoration_y).sum().item()
        correct['vessel'] += (vessel_out.argmax(1) == vessel_y).sum().item()
        total += period_y.size(0)

        # Update progress bar
        pbar.set_postfix({'loss': f'{total_loss/(pbar.n+1):.3f}'})

    return (total_loss / len(loader),
            {k: v/total for k, v in correct.items()})

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = {'period': 0, 'decoration': 0, 'vessel': 0}
    total = 0

    all_preds = {'period': [], 'decoration': [], 'vessel': []}
    all_labels = {'period': [], 'decoration': [], 'vessel': []}

    with torch.no_grad():
        for images, period_y, decoration_y, vessel_y in loader:
            images = images.to(device)
            period_y = period_y.to(device)
            decoration_y = decoration_y.to(device)
            vessel_y = vessel_y.to(device)

            period_out, decoration_out, vessel_out = model(images)

            loss = (criterion(period_out, period_y) +
                    criterion(decoration_out, decoration_y) +
                    criterion(vessel_out, vessel_y))

            total_loss += loss.item()

            period_pred = period_out.argmax(1)
            decoration_pred = decoration_out.argmax(1)
            vessel_pred = vessel_out.argmax(1)

            correct['period'] += (period_pred == period_y).sum().item()
            correct['decoration'] += (decoration_pred == decoration_y).sum().item()
            correct['vessel'] += (vessel_pred == vessel_y).sum().item()
            total += period_y.size(0)

            all_preds['period'].extend(period_pred.cpu().numpy())
            all_preds['decoration'].extend(decoration_pred.cpu().numpy())
            all_preds['vessel'].extend(vessel_pred.cpu().numpy())
            all_labels['period'].extend(period_y.cpu().numpy())
            all_labels['decoration'].extend(decoration_y.cpu().numpy())
            all_labels['vessel'].extend(vessel_y.cpu().numpy())

    return (total_loss / len(loader),
            {k: v/total for k, v in correct.items()},
            all_preds, all_labels)

# ============================================
# CROSS-VALIDATION
# ============================================
print(f"\n[2/7] Starting {N_FOLDS}-Fold Cross-Validation...")

# Create stratification labels (combined)
strat_labels = (df['period_encoded'] * 1000 +
                df['decoration_encoded'] * 100 +
                df['vessel_encoded'])

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

cv_results = {
    'train_loss': [], 'val_loss': [],
    'period_train': [], 'period_val': [],
    'decoration_train': [], 'decoration_val': [],
    'vessel_train': [], 'vessel_val': [],
    'period_f1': [], 'decoration_f1': [], 'vessel_f1': []
}

fold_histories = []

print(f"\n   Starting cross-validation...")
fold_pbar = tqdm(enumerate(skf.split(df, strat_labels)), total=N_FOLDS, desc="📊 CV Folds", ncols=100)
for fold, (train_idx, val_idx) in fold_pbar:
    fold_pbar.set_description(f"📊 Fold {fold+1}/{N_FOLDS}")

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    tqdm.write(f"\n   Fold {fold+1}: Train={len(train_df)} | Val={len(val_df)}")

    train_dataset = CeramicDataset(train_df, train_transform)
    val_dataset = CeramicDataset(val_df, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CeramicClassifierV2(n_period, n_decoration, n_vessel).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_val_acc = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    epoch_pbar = tqdm(range(EPOCHS), desc=f"  🔄 Epochs", leave=False, ncols=100)
    for epoch in epoch_pbar:
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, f"    Training")
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        avg_train_acc = np.mean(list(train_acc.values()))
        avg_val_acc = np.mean(list(val_acc.values()))

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)

        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'loss': f'{val_loss:.3f}',
            'acc': f'{avg_val_acc:.3f}',
            'best': f'{best_val_acc:.3f}'
        })

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            tqdm.write(f"   ⏹️  Early stopping at epoch {epoch+1}")
            break

    fold_histories.append(history)

    # Load best model for final eval
    model.load_state_dict(best_model_state)
    val_loss, val_acc, preds, labels = evaluate(model, val_loader, criterion, device)

    # Calculate F1 scores
    f1_period = f1_score(labels['period'], preds['period'], average='weighted')
    f1_decoration = f1_score(labels['decoration'], preds['decoration'], average='weighted')
    f1_vessel = f1_score(labels['vessel'], preds['vessel'], average='weighted')

    cv_results['val_loss'].append(val_loss)
    cv_results['period_val'].append(val_acc['period'])
    cv_results['decoration_val'].append(val_acc['decoration'])
    cv_results['vessel_val'].append(val_acc['vessel'])
    cv_results['period_f1'].append(f1_period)
    cv_results['decoration_f1'].append(f1_decoration)
    cv_results['vessel_f1'].append(f1_vessel)

    tqdm.write(f"   ✅ Fold {fold+1} Results:")
    tqdm.write(f"      Period:     Acc={val_acc['period']:.4f} | F1={f1_period:.4f}")
    tqdm.write(f"      Decoration: Acc={val_acc['decoration']:.4f} | F1={f1_decoration:.4f}")
    tqdm.write(f"      Vessel:     Acc={val_acc['vessel']:.4f} | F1={f1_vessel:.4f}")

# ============================================
# CROSS-VALIDATION SUMMARY
# ============================================
print("\n" + "=" * 70)
print("   CROSS-VALIDATION SUMMARY")
print("=" * 70)

print(f"\n   {'Task':<15} {'Accuracy':<20} {'F1-Score':<20}")
print(f"   {'-'*55}")
print(f"   {'Period':<15} {np.mean(cv_results['period_val']):.4f} ± {np.std(cv_results['period_val']):.4f}    "
      f"{np.mean(cv_results['period_f1']):.4f} ± {np.std(cv_results['period_f1']):.4f}")
print(f"   {'Decoration':<15} {np.mean(cv_results['decoration_val']):.4f} ± {np.std(cv_results['decoration_val']):.4f}    "
      f"{np.mean(cv_results['decoration_f1']):.4f} ± {np.std(cv_results['decoration_f1']):.4f}")
print(f"   {'Vessel Type':<15} {np.mean(cv_results['vessel_val']):.4f} ± {np.std(cv_results['vessel_val']):.4f}    "
      f"{np.mean(cv_results['vessel_f1']):.4f} ± {np.std(cv_results['vessel_f1']):.4f}")

# ============================================
# OVERFITTING ANALYSIS
# ============================================
print("\n[3/7] Overfitting Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot training curves for all folds
for i, history in enumerate(fold_histories):
    axes[0, 0].plot(history['train_loss'], alpha=0.7, label=f'Fold {i+1}' if i == 0 else '')
    axes[0, 1].plot(history['val_loss'], alpha=0.7)
    axes[1, 0].plot(history['train_acc'], alpha=0.7)
    axes[1, 1].plot(history['val_acc'], alpha=0.7)

axes[0, 0].set_title('Training Loss per Fold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 1].set_title('Validation Loss per Fold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[1, 0].set_title('Training Accuracy per Fold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 1].set_title('Validation Accuracy per Fold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'training_curves.png'), dpi=150)
print(f"   Saved: training_curves.png")

# Overfitting metric: gap between train and val accuracy
print("\n   Overfitting Analysis (Train-Val Gap):")
for i, history in enumerate(fold_histories):
    final_train = history['train_acc'][-1]
    final_val = history['val_acc'][-1]
    gap = final_train - final_val
    status = "OK" if gap < 0.1 else "WARNING" if gap < 0.2 else "OVERFITTING"
    print(f"      Fold {i+1}: Train={final_train:.4f} Val={final_val:.4f} Gap={gap:.4f} [{status}]")

# ============================================
# TRAIN FINAL MODEL
# ============================================
print("\n[4/7] Training final model on all data...")

full_dataset = CeramicDataset(df, train_transform)
full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

final_model = CeramicClassifierV2(n_period, n_decoration, n_vessel).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

final_pbar = tqdm(range(EPOCHS), desc="🎯 Final Training", ncols=100)
for epoch in final_pbar:
    train_loss, train_acc = train_epoch(final_model, full_loader, criterion, optimizer, device, "  Training")
    scheduler.step()

    final_pbar.set_postfix({
        'loss': f'{train_loss:.3f}',
        'P': f'{train_acc["period"]:.2f}',
        'D': f'{train_acc["decoration"]:.2f}',
        'V': f'{train_acc["vessel"]:.2f}'
    })

# ============================================
# FINAL EVALUATION & CONFUSION MATRICES
# ============================================
print("\n[5/7] Final Evaluation & Confusion Matrices...")

eval_dataset = CeramicDataset(df, val_transform)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

_, final_acc, all_preds, all_labels = evaluate(final_model, eval_loader, criterion, device)

print(f"\n   Final Model Accuracy:")
print(f"      Period:     {final_acc['period']:.4f}")
print(f"      Decoration: {final_acc['decoration']:.4f}")
print(f"      Vessel:     {final_acc['vessel']:.4f}")

# Classification reports
print("\n" + "=" * 70)
print("   CLASSIFICATION REPORTS")
print("=" * 70)

tasks = [
    ('Period', all_labels['period'], all_preds['period'], le_period.classes_),
    ('Decoration', all_labels['decoration'], all_preds['decoration'], le_decoration.classes_),
    ('Vessel Type', all_labels['vessel'], all_preds['vessel'], le_vessel.classes_)
]

for name, labels, preds, classes in tasks:
    print(f"\n   {name}:")
    print(classification_report(labels, preds, target_names=classes, zero_division=0))

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, labels, preds, classes) in zip(axes, tasks):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(f'{name} Confusion Matrix')
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrices.png'), dpi=150)
print(f"\n   Saved: confusion_matrices.png")

# ============================================
# FALSE POSITIVES/NEGATIVES ANALYSIS
# ============================================
print("\n[6/7] False Positives/Negatives Analysis...")

for name, labels, preds, classes in tasks:
    print(f"\n   {name} - Misclassifications:")

    errors = []
    for i, (true, pred) in enumerate(zip(labels, preds)):
        if true != pred:
            errors.append({
                'true': classes[true],
                'pred': classes[pred],
                'id': df.iloc[i]['id']
            })

    # Group by confusion pair
    confusion_pairs = {}
    for e in errors:
        key = f"{e['true']} -> {e['pred']}"
        if key not in confusion_pairs:
            confusion_pairs[key] = []
        confusion_pairs[key].append(e['id'])

    # Sort by frequency
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: -len(x[1]))

    print(f"      Total errors: {len(errors)} / {len(labels)} ({len(errors)/len(labels)*100:.1f}%)")
    print(f"      Top confusion pairs:")
    for pair, ids in sorted_pairs[:5]:
        print(f"         {pair}: {len(ids)} cases")

# ============================================
# SAVE MODEL AND ARTIFACTS
# ============================================
print("\n[7/7] Saving model and artifacts...")

model_path = os.path.join(MODEL_DIR, "ceramic_classifier_v2.pt")
torch.save({
    'model_state_dict': final_model.state_dict(),
    'n_period': n_period,
    'n_decoration': n_decoration,
    'n_vessel': n_vessel,
}, model_path)
print(f"   Model: {model_path}")

encoders = {
    'period_classes': le_period.classes_.tolist(),
    'decoration_classes': le_decoration.classes_.tolist(),
    'vessel_classes': le_vessel.classes_.tolist()
}
encoders_path = os.path.join(MODEL_DIR, "label_encoders_v2.json")
with open(encoders_path, 'w') as f:
    json.dump(encoders, f, indent=2)
print(f"   Encoders: {encoders_path}")

# Training report
report = {
    'training_date': datetime.now().isoformat(),
    'framework': 'PyTorch',
    'model': 'ResNet50 + Multi-head',
    'total_images': len(df),
    'image_size': IMG_SIZE,
    'batch_size': BATCH_SIZE,
    'epochs': EPOCHS,
    'n_folds': N_FOLDS,
    'learning_rate': LEARNING_RATE,
    'weight_decay': WEIGHT_DECAY,
    'device': str(device),
    'cross_validation': {
        'period': {
            'accuracy': f"{np.mean(cv_results['period_val']):.4f} ± {np.std(cv_results['period_val']):.4f}",
            'f1_score': f"{np.mean(cv_results['period_f1']):.4f} ± {np.std(cv_results['period_f1']):.4f}"
        },
        'decoration': {
            'accuracy': f"{np.mean(cv_results['decoration_val']):.4f} ± {np.std(cv_results['decoration_val']):.4f}",
            'f1_score': f"{np.mean(cv_results['decoration_f1']):.4f} ± {np.std(cv_results['decoration_f1']):.4f}"
        },
        'vessel_type': {
            'accuracy': f"{np.mean(cv_results['vessel_val']):.4f} ± {np.std(cv_results['vessel_val']):.4f}",
            'f1_score': f"{np.mean(cv_results['vessel_f1']):.4f} ± {np.std(cv_results['vessel_f1']):.4f}"
        }
    },
    'final_model': {
        'period_accuracy': float(final_acc['period']),
        'decoration_accuracy': float(final_acc['decoration']),
        'vessel_accuracy': float(final_acc['vessel'])
    },
    'class_distribution': {
        'period': df['macro_period'].value_counts().to_dict(),
        'decoration': df['decoration'].value_counts().to_dict(),
        'vessel_type': df['vessel_type'].value_counts().to_dict()
    }
}

report_path = os.path.join(MODEL_DIR, "training_report_v2.json")
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"   Report: {report_path}")

print("\n" + "=" * 70)
print("   TRAINING COMPLETE!")
print("=" * 70)
print(f"\n   Output directory: {MODEL_DIR}")
print(f"\n   Files generated:")
print(f"      - ceramic_classifier_v2.pt")
print(f"      - label_encoders_v2.json")
print(f"      - training_report_v2.json")
print(f"      - training_curves.png")
print(f"      - confusion_matrices.png")
print("=" * 70)
