#!/usr/bin/env python3
"""
Ceramic Classification ML Model - PyTorch Version
CNN classifier with cross-validation for archaeological ceramics
Classifies: macro_period and decoration (plain/decorated)
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

print("=" * 60)
print("    CERAMIC CLASSIFICATION ML MODEL TRAINING")
print("    Using PyTorch CNN")
print("=" * 60)

# Check/install packages
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    import torchvision.models as models
    from PIL import Image
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import cv2
except ImportError as e:
    print(f"Installing missing packages...")
    os.system("pip install torch torchvision scikit-learn opencv-python-headless pillow")
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    import torchvision.models as models
    from PIL import Image
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import cv2

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25
N_FOLDS = 5
LEARNING_RATE = 0.0001
DESKTOP_PATH = os.path.expanduser("~/Desktop")
MODEL_DIR = os.path.join(DESKTOP_PATH, "CeramicaML_Model")

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")

# Create output directory
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"Model will be saved to: {MODEL_DIR}")

# Load data from SQLite
print("\n[1/6] Loading data from database...")
conn = sqlite3.connect('ceramica.db')
df = pd.read_sql_query("""
    SELECT id, collection, macro_period, decoration, image_path
    FROM items
    WHERE macro_period IS NOT NULL AND macro_period != ''
    AND decoration IS NOT NULL AND decoration != ''
""", conn)
conn.close()

print(f"   Loaded {len(df)} items with complete data")

# Encode labels
le_period = LabelEncoder()
le_decoration = LabelEncoder()

df['period_encoded'] = le_period.fit_transform(df['macro_period'])
df['decoration_encoded'] = le_decoration.fit_transform(df['decoration'])

n_period_classes = len(le_period.classes_)
n_decoration_classes = len(le_decoration.classes_)

print(f"   Period classes ({n_period_classes}): {list(le_period.classes_)}")
print(f"   Decoration classes ({n_decoration_classes}): {list(le_decoration.classes_)}")

# Dataset class
class CeramicDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        try:
            img = Image.open(row['image_path']).convert('RGB')
        except:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))

        if self.transform:
            img = self.transform(img)

        period_label = torch.tensor(row['period_encoded'], dtype=torch.long)
        decoration_label = torch.tensor(row['decoration_encoded'], dtype=torch.long)

        return img, period_label, decoration_label

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Multi-output CNN Model
class CeramicClassifier(nn.Module):
    def __init__(self, n_period_classes, n_decoration_classes):
        super(CeramicClassifier, self).__init__()

        # Load pretrained ResNet18 (smaller, faster)
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Replace the final FC layer
        n_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Shared features
        self.shared = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Classification heads
        self.period_head = nn.Linear(256, n_period_classes)
        self.decoration_head = nn.Linear(256, n_decoration_classes)

    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared(features)
        period_out = self.period_head(shared)
        decoration_out = self.decoration_head(shared)
        return period_out, decoration_out

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    period_correct = 0
    decoration_correct = 0
    total = 0

    for images, period_labels, decoration_labels in loader:
        images = images.to(device)
        period_labels = period_labels.to(device)
        decoration_labels = decoration_labels.to(device)

        optimizer.zero_grad()

        period_out, decoration_out = model(images)

        loss_period = criterion(period_out, period_labels)
        loss_decoration = criterion(decoration_out, decoration_labels)
        loss = loss_period + loss_decoration

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, period_pred = torch.max(period_out, 1)
        _, decoration_pred = torch.max(decoration_out, 1)

        period_correct += (period_pred == period_labels).sum().item()
        decoration_correct += (decoration_pred == decoration_labels).sum().item()
        total += period_labels.size(0)

    return total_loss / len(loader), period_correct / total, decoration_correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    period_correct = 0
    decoration_correct = 0
    total = 0

    all_period_preds = []
    all_period_labels = []
    all_decoration_preds = []
    all_decoration_labels = []

    with torch.no_grad():
        for images, period_labels, decoration_labels in loader:
            images = images.to(device)
            period_labels = period_labels.to(device)
            decoration_labels = decoration_labels.to(device)

            period_out, decoration_out = model(images)

            loss_period = criterion(period_out, period_labels)
            loss_decoration = criterion(decoration_out, decoration_labels)
            loss = loss_period + loss_decoration

            total_loss += loss.item()
            _, period_pred = torch.max(period_out, 1)
            _, decoration_pred = torch.max(decoration_out, 1)

            period_correct += (period_pred == period_labels).sum().item()
            decoration_correct += (decoration_pred == decoration_labels).sum().item()
            total += period_labels.size(0)

            all_period_preds.extend(period_pred.cpu().numpy())
            all_period_labels.extend(period_labels.cpu().numpy())
            all_decoration_preds.extend(decoration_pred.cpu().numpy())
            all_decoration_labels.extend(decoration_labels.cpu().numpy())

    return (total_loss / len(loader),
            period_correct / total,
            decoration_correct / total,
            all_period_preds, all_period_labels,
            all_decoration_preds, all_decoration_labels)

# Cross-validation
print(f"\n[2/6] Starting {N_FOLDS}-Fold Cross-Validation...")

combined_labels = df['period_encoded'] * n_decoration_classes + df['decoration_encoded']
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

cv_results = {
    'period_val_accuracy': [],
    'decoration_val_accuracy': []
}

for fold, (train_idx, val_idx) in enumerate(skf.split(df, combined_labels)):
    print(f"\n{'='*50}")
    print(f"   FOLD {fold + 1}/{N_FOLDS}")
    print(f"{'='*50}")
    print(f"   Train: {len(train_idx)} | Validation: {len(val_idx)}")

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_dataset = CeramicDataset(train_df, train_transform)
    val_dataset = CeramicDataset(val_df, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Create model
    model = CeramicClassifier(n_period_classes, n_decoration_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_acc = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_loss, train_period_acc, train_dec_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_period_acc, val_dec_acc, _, _, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        avg_val_acc = (val_period_acc + val_dec_acc) / 2

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}: Loss={train_loss:.4f} | Period={val_period_acc:.3f} | Decor={val_dec_acc:.3f}")

        if patience_counter >= 5:
            print(f"   Early stopping at epoch {epoch+1}")
            break

    # Final evaluation
    _, val_period_acc, val_dec_acc, _, _, _, _ = evaluate(model, val_loader, criterion, device)
    cv_results['period_val_accuracy'].append(val_period_acc)
    cv_results['decoration_val_accuracy'].append(val_dec_acc)

    print(f"   Final - Period: {val_period_acc:.4f} | Decoration: {val_dec_acc:.4f}")

# Summary
print("\n" + "=" * 60)
print("   CROSS-VALIDATION SUMMARY")
print("=" * 60)
print(f"\n   Period Classification:")
print(f"      Mean Accuracy: {np.mean(cv_results['period_val_accuracy']):.4f}")
print(f"      Std Deviation: {np.std(cv_results['period_val_accuracy']):.4f}")
print(f"      Per-fold: {[f'{x:.4f}' for x in cv_results['period_val_accuracy']]}")

print(f"\n   Decoration Classification:")
print(f"      Mean Accuracy: {np.mean(cv_results['decoration_val_accuracy']):.4f}")
print(f"      Std Deviation: {np.std(cv_results['decoration_val_accuracy']):.4f}")
print(f"      Per-fold: {[f'{x:.4f}' for x in cv_results['decoration_val_accuracy']]}")

# Train final model on all data
print("\n[3/6] Training final model on all data...")

full_dataset = CeramicDataset(df, train_transform)
full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

final_model = CeramicClassifier(n_period_classes, n_decoration_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(final_model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    train_loss, period_acc, dec_acc = train_epoch(final_model, full_loader, criterion, optimizer, device)
    if (epoch + 1) % 5 == 0:
        print(f"   Epoch {epoch+1}/{EPOCHS}: Loss={train_loss:.4f} | Period={period_acc:.3f} | Decor={dec_acc:.3f}")

# Final evaluation
print("\n[4/6] Final Model Evaluation...")
eval_dataset = CeramicDataset(df, val_transform)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

_, final_period_acc, final_dec_acc, period_preds, period_labels, dec_preds, dec_labels = evaluate(
    final_model, eval_loader, criterion, device
)

print(f"   Final Period Accuracy: {final_period_acc:.4f}")
print(f"   Final Decoration Accuracy: {final_dec_acc:.4f}")

print("\n   Period Classification Report:")
print(classification_report(period_labels, period_preds, target_names=le_period.classes_))

print("\n   Decoration Classification Report:")
print(classification_report(dec_labels, dec_preds, target_names=le_decoration.classes_))

# Save model
print("\n[5/6] Saving model and artifacts...")

model_path = os.path.join(MODEL_DIR, "ceramic_classifier.pt")
torch.save({
    'model_state_dict': final_model.state_dict(),
    'n_period_classes': n_period_classes,
    'n_decoration_classes': n_decoration_classes,
}, model_path)
print(f"   Model saved: {model_path}")

# Save label encoders
encoders = {
    'period_classes': le_period.classes_.tolist(),
    'decoration_classes': le_decoration.classes_.tolist()
}
encoders_path = os.path.join(MODEL_DIR, "label_encoders.json")
with open(encoders_path, 'w') as f:
    json.dump(encoders, f, indent=2)
print(f"   Encoders saved: {encoders_path}")

# Save training report
report = {
    'training_date': datetime.now().isoformat(),
    'framework': 'PyTorch',
    'model': 'ResNet18 + Multi-head',
    'total_images': len(df),
    'image_size': IMG_SIZE,
    'n_folds': N_FOLDS,
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'device': str(device),
    'cross_validation': {
        'period_mean_accuracy': float(np.mean(cv_results['period_val_accuracy'])),
        'period_std': float(np.std(cv_results['period_val_accuracy'])),
        'decoration_mean_accuracy': float(np.mean(cv_results['decoration_val_accuracy'])),
        'decoration_std': float(np.std(cv_results['decoration_val_accuracy'])),
    },
    'final_model': {
        'period_accuracy': float(final_period_acc),
        'decoration_accuracy': float(final_dec_acc)
    },
    'class_distribution': {
        'periods': df['macro_period'].value_counts().to_dict(),
        'decorations': df['decoration'].value_counts().to_dict()
    }
}

report_path = os.path.join(MODEL_DIR, "training_report.json")
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"   Report saved: {report_path}")

# Save inference script
inference_script = '''#!/usr/bin/env python3
"""
Ceramic Classification - Inference Script
Usage: python predict_ceramic.py <image_path>
"""
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = "ceramic_classifier.pt"
ENCODERS_PATH = "label_encoders.json"
IMG_SIZE = 224

class CeramicClassifier(nn.Module):
    def __init__(self, n_period_classes, n_decoration_classes):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        n_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.shared = nn.Sequential(nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(0.3))
        self.period_head = nn.Linear(256, n_period_classes)
        self.decoration_head = nn.Linear(256, n_decoration_classes)

    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared(features)
        return self.period_head(shared), self.decoration_head(shared)

def predict(image_path):
    with open(ENCODERS_PATH) as f:
        encoders = json.load(f)

    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    model = CeramicClassifier(checkpoint['n_period_classes'], checkpoint['n_decoration_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        period_out, decoration_out = model(img_tensor)
        period_probs = torch.softmax(period_out, dim=1)
        decoration_probs = torch.softmax(decoration_out, dim=1)

    period_idx = period_probs.argmax().item()
    decoration_idx = decoration_probs.argmax().item()

    print(f"Macro Period: {encoders['period_classes'][period_idx]} ({period_probs[0][period_idx]*100:.1f}%)")
    print(f"Decoration: {encoders['decoration_classes'][decoration_idx]} ({decoration_probs[0][decoration_idx]*100:.1f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_ceramic.py <image_path>")
    else:
        predict(sys.argv[1])
'''

inference_path = os.path.join(MODEL_DIR, "predict_ceramic.py")
with open(inference_path, 'w') as f:
    f.write(inference_script)
print(f"   Inference script saved: {inference_path}")

print("\n" + "=" * 60)
print("   TRAINING COMPLETE!")
print("=" * 60)
print(f"\n   Model saved to: {MODEL_DIR}")
print(f"\n   To use the model:")
print(f"      cd '{MODEL_DIR}'")
print(f"      python predict_ceramic.py <path_to_image>")
print("=" * 60)
