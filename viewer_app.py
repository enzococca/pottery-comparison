#!/usr/bin/env python3
"""
CeramicaDatabase - Unified Multi-Collection Viewer
Archaeological ceramic viewer with SQLite database, user roles, and ML API

Run with: python viewer_app.py
"""

import os
import sys
import json
import subprocess
import webbrowser
import sqlite3
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import pandas as pd
import re
import cv2
import numpy as np
import hashlib
import secrets
import http.cookies
from datetime import datetime
import base64
import io

# PDF Scale extraction (lazy loaded)
def extract_scale_from_pdf(pdf_path, page_num=None):
    """
    Extract scale information from PDF pages.
    Looks for patterns like:
    - "1:3", "scala 1:3", "scale 1:3"
    - "échelle 1:3"
    - "5 cm" with scale bar annotations
    Returns list of found scales with page numbers.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return {'error': 'PyMuPDF not installed. Run: pip install pymupdf'}

    if not os.path.exists(pdf_path):
        return {'error': f'PDF not found: {pdf_path}'}

    scales_found = []
    scale_patterns = [
        r'(?:scala|scale|échelle|escala)?\s*(\d+)\s*:\s*(\d+)',  # 1:3, scala 1:3
        r'(\d+(?:[.,]\d+)?)\s*(?:cm|mm|m)\b',  # 5 cm, 10 mm (scale bar labels)
    ]

    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        pages_to_check = [page_num - 1] if page_num else range(total_pages)

        for page_idx in pages_to_check:
            if page_idx < 0 or page_idx >= total_pages:
                continue

            page = doc[page_idx]
            text = page.get_text()

            # Look for scale patterns
            for pattern in scale_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple) and len(match) == 2:
                        try:
                            num, denom = int(match[0]), int(match[1])
                            if 1 <= num <= 10 and 1 <= denom <= 10:
                                scale_str = f"{num}:{denom}"
                                if scale_str not in [s['scale'] for s in scales_found]:
                                    scales_found.append({
                                        'scale': scale_str,
                                        'page': page_idx + 1,
                                        'ratio': num / denom
                                    })
                        except:
                            pass

        doc.close()
        return {'scales': scales_found, 'total_pages': total_pages}

    except Exception as e:
        return {'error': str(e)}


# ML Model imports (lazy loaded)
ML_MODEL = None
ML_ENCODERS = None
ML_TRANSFORM = None

# Image Similarity Search
EMBEDDINGS = None
EMBEDDINGS_METADATA = None
FEATURE_EXTRACTOR = None

# Configuration
PORT = int(os.environ.get('PORT', 8080))
HOST = os.environ.get('HOST', '0.0.0.0')
IS_RAILWAY = os.environ.get('RAILWAY_ENVIRONMENT') is not None

# Database - Use DATA_DIR for persistent storage on Railway
DATA_DIR = os.environ.get('DATA_DIR', '')  # Set to /data on Railway with volume
DB_FILE = os.path.join(DATA_DIR, "ceramica.db") if DATA_DIR else "ceramica.db"
CSV_FILE = "ceramica_metadata.csv"
CONFIG_FILE = "config.json"

# Authentication - Password hashes (SHA256)
# Default admin password: admin2024
ADMIN_HASH = os.environ.get('ADMIN_HASH', 'b8b8eb83374c0bf3b1c3224159f6119dbfff1b7ed6dfecdd80d4e8a895790a34')
# Default viewer password: viewer2024
VIEWER_HASH = os.environ.get('VIEWER_HASH', '292a886d8da982974b3e9ad1ad61c0328f075ee17cd0eb0a3aba3aa03481a3b9')

# Session storage: {token: {'role': 'admin'|'viewer', 'created': timestamp}}
SESSIONS = {}

# Macro-period mappings
MACRO_PERIODS = {
    "Umm an-Nar": ["umm an-nar", "umm-an-nar", "hili", "2700-2000", "2500-2000",
                   "iii millennio", "iiie millénaire", "prima eta del bronzo", "early bronze", "hili ii"],
    "Wadi Suq": ["wadi suq", "wadi-suq", "2000-1600", "2000-1800", "1800-1600", "bronze moyen", "middle bronze"],
    "Late Bronze Age": ["bronze récent", "bronze recent", "late bronze", "1600-1250",
                        "1600-600", "fer i", "fer ii", "iron age", "fer ", "masafi"]
}

# ML Model paths
ML_MODEL_DIR = Path(__file__).parent / "ml_model"
ML_MODEL_PATH = ML_MODEL_DIR / "ceramic_classifier_v2.pt"
ML_ENCODERS_PATH = ML_MODEL_DIR / "label_encoders_v2.json"


def load_ml_model():
    """Load ML model v2 lazily - Multi-task with Period, Decoration, Vessel Type"""
    global ML_MODEL, ML_ENCODERS, ML_TRANSFORM

    if ML_MODEL is not None:
        return True

    try:
        import torch
        import torch.nn as nn
        from torchvision import transforms, models
        from PIL import Image

        # Define model architecture v2 - ResNet50 + 3 heads
        class CeramicClassifierV2(nn.Module):
            def __init__(self, n_period, n_decoration, n_vessel, dropout=0.4):
                super().__init__()
                self.backbone = models.resnet50(weights=None)
                n_features = self.backbone.fc.in_features  # 2048
                self.backbone.fc = nn.Identity()

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

                self.period_head = nn.Sequential(
                    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout/2), nn.Linear(128, n_period)
                )
                self.decoration_head = nn.Sequential(
                    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout/2), nn.Linear(128, n_decoration)
                )
                self.vessel_head = nn.Sequential(
                    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout/2), nn.Linear(128, n_vessel)
                )

            def forward(self, x):
                features = self.backbone(x)
                shared = self.shared(features)
                return self.period_head(shared), self.decoration_head(shared), self.vessel_head(shared)

        # Load encoders
        with open(ML_ENCODERS_PATH) as f:
            ML_ENCODERS = json.load(f)

        # Load model
        checkpoint = torch.load(ML_MODEL_PATH, map_location='cpu', weights_only=True)
        ML_MODEL = CeramicClassifierV2(
            checkpoint['n_period'],
            checkpoint['n_decoration'],
            checkpoint['n_vessel']
        )
        ML_MODEL.load_state_dict(checkpoint['model_state_dict'])
        ML_MODEL.eval()

        # Create transform
        ML_TRANSFORM = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        print("ML Model v2 loaded successfully (Period + Decoration + Vessel Type)")
        return True

    except Exception as e:
        print(f"Error loading ML model: {e}")
        import traceback
        traceback.print_exc()
        return False


def photo_to_drawing(image_data, threshold=18, min_area_pct=0.05, blur_size=9):
    """
    Convert a real photo to a drawing-like image.
    v6: Configurable parameters for decoration extraction.

    Args:
        image_data: Base64 encoded image
        threshold: Contrast threshold for decoration detection (default 18)
        min_area_pct: Minimum area percentage for noise filtering (default 0.05)
        blur_size: Bilateral filter size (default 9)

    Returns base64 encoded processed image.
    """
    try:
        from PIL import Image

        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_array = np.array(img)

        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Step 1: Create mask for non-black areas (the ceramic object)
        _, object_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        kernel = np.ones((7, 7), np.uint8)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)

        # Step 2: Blur to reduce texture (configurable)
        blur_size = max(1, blur_size if blur_size % 2 == 1 else blur_size + 1)  # Must be odd
        gray_smooth = cv2.bilateralFilter(gray, blur_size, 50, 50)
        gray_smooth = cv2.GaussianBlur(gray_smooth, (5, 5), 0)

        # Step 3: Find decorations using local contrast (configurable threshold)
        local_mean = cv2.blur(gray_smooth, (41, 41))
        decoration_mask = (gray_smooth < (local_mean - threshold)).astype(np.uint8) * 255

        # Step 4: Remove small noise using connected components (configurable min area)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(decoration_mask, connectivity=8)
        min_area = int(gray.shape[0] * gray.shape[1] * (min_area_pct / 100.0))

        # Create cleaned decoration mask
        decoration_clean = np.zeros_like(decoration_mask)
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                decoration_clean[labels == i] = 255

        # Step 5: Morphological cleanup - close small gaps, remove remaining noise
        kernel_morph = np.ones((5, 5), np.uint8)
        decoration_clean = cv2.morphologyEx(decoration_clean, cv2.MORPH_CLOSE, kernel_morph)
        decoration_clean = cv2.morphologyEx(decoration_clean, cv2.MORPH_OPEN, kernel_morph)

        # Step 6: Create result - white background, black decorations
        result_gray = np.ones_like(gray) * 255
        result_gray[decoration_clean > 0] = 0

        # Step 7: Apply object mask - only show within ceramic area
        result_gray = np.where(object_mask > 0, result_gray, 255).astype(np.uint8)

        # Step 8: Add clean outline of the ceramic object
        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_gray, contours, -1, 0, 2)

        # Convert to 3-channel for consistency
        result = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2RGB)

        # Encode back to base64
        result_pil = Image.fromarray(result)
        buffer = io.BytesIO()
        result_pil.save(buffer, format='PNG')
        result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return {
            'success': True,
            'processed_image': f'data:image/png;base64,{result_base64}',
            'original_size': img.size,
            'method': 'decoration_extraction_v5'
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def combine_drawing_with_contour(original_image, drawing_image, preprocessed_image=None):
    """
    Combine user's manual decoration drawing with automatically extracted ceramic contour.

    Args:
        original_image: Base64 encoded original photo
        drawing_image: Base64 encoded user drawing (black lines on transparent/white)
        preprocessed_image: Optional base64 encoded preprocessed image (with user's parameter settings)

    If preprocessed_image is provided, it's used as the base (already has contour + auto-extracted decorations).
    Otherwise, contour is extracted from original_image.

    Returns base64 encoded combined image.
    """
    try:
        from PIL import Image

        # Decode original image (needed for dimensions)
        if ',' in original_image:
            original_image = original_image.split(',')[1]
        original_bytes = base64.b64decode(original_image)
        original = Image.open(io.BytesIO(original_bytes)).convert('RGB')
        original_array = np.array(original)

        # Decode drawing image
        if ',' in drawing_image:
            drawing_image = drawing_image.split(',')[1]
        drawing_bytes = base64.b64decode(drawing_image)
        drawing = Image.open(io.BytesIO(drawing_bytes)).convert('RGBA')
        drawing_array = np.array(drawing)

        # If preprocessed image is provided, use it as the base
        if preprocessed_image:
            if ',' in preprocessed_image:
                preprocessed_image = preprocessed_image.split(',')[1]
            preprocessed_bytes = base64.b64decode(preprocessed_image)
            preprocessed = Image.open(io.BytesIO(preprocessed_bytes)).convert('RGB')
            preprocessed_array = np.array(preprocessed)

            # Convert to grayscale - preprocessed is white bg with black decorations/contour
            result = cv2.cvtColor(preprocessed_array, cv2.COLOR_RGB2GRAY)
        else:
            # Extract contour from original image
            gray = cv2.cvtColor(original_array, cv2.COLOR_RGB2GRAY)

            # Create mask for non-black areas (the ceramic object)
            _, object_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
            kernel = np.ones((7, 7), np.uint8)
            object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
            object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)

            # Create result image - white background
            result = np.ones((original_array.shape[0], original_array.shape[1]), dtype=np.uint8) * 255

            # Extract and draw ceramic contour
            contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, 0, 2)

        # Resize drawing to match result if needed
        if drawing_array.shape[:2] != result.shape[:2]:
            drawing_pil = Image.fromarray(drawing_array)
            drawing_pil = drawing_pil.resize((result.shape[1], result.shape[0]), Image.Resampling.LANCZOS)
            drawing_array = np.array(drawing_pil)

        # Extract drawing marks (look for non-white, non-transparent pixels)
        # The drawing has RGBA - alpha channel indicates drawn areas
        if drawing_array.shape[2] == 4:
            # Use alpha channel to find drawn areas
            alpha = drawing_array[:, :, 3]
            # Also check RGB - drawn areas are typically dark
            rgb_dark = np.mean(drawing_array[:, :, :3], axis=2) < 200
            drawing_mask = (alpha > 50) & rgb_dark
        else:
            # Fallback: just look for dark pixels
            drawing_mask = np.mean(drawing_array[:, :, :3], axis=2) < 128

        # Apply drawing to result (black where user drew)
        result[drawing_mask] = 0

        # Convert to 3-channel for consistency
        result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

        # Encode back to base64
        result_pil = Image.fromarray(result_rgb)
        buffer = io.BytesIO()
        result_pil.save(buffer, format='PNG')
        result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return {
            'success': True,
            'combined_image': f'data:image/png;base64,{result_base64}',
            'original_size': original.size
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def classify_image(image_data):
    """Classify an image using the ML model v2 - Period, Decoration, Vessel Type"""
    if not load_ml_model():
        return {'error': 'ML model not available'}

    try:
        import torch
        from PIL import Image

        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Transform and predict
        img_tensor = ML_TRANSFORM(img).unsqueeze(0)

        with torch.no_grad():
            period_out, decoration_out, vessel_out = ML_MODEL(img_tensor)
            period_probs = torch.softmax(period_out, dim=1)
            decoration_probs = torch.softmax(decoration_out, dim=1)
            vessel_probs = torch.softmax(vessel_out, dim=1)

        # Get all predictions with probabilities
        period_results = []
        for i, prob in enumerate(period_probs[0]):
            period_results.append({
                'class': ML_ENCODERS['period_classes'][i],
                'confidence': float(prob) * 100
            })
        period_results.sort(key=lambda x: -x['confidence'])

        decoration_results = []
        for i, prob in enumerate(decoration_probs[0]):
            decoration_results.append({
                'class': ML_ENCODERS['decoration_classes'][i],
                'confidence': float(prob) * 100
            })
        decoration_results.sort(key=lambda x: -x['confidence'])

        vessel_results = []
        for i, prob in enumerate(vessel_probs[0]):
            vessel_results.append({
                'class': ML_ENCODERS['vessel_classes'][i],
                'confidence': float(prob) * 100
            })
        vessel_results.sort(key=lambda x: -x['confidence'])

        return {
            'success': True,
            'period': period_results,
            'decoration': decoration_results,
            'vessel_type': vessel_results,
            'predicted_period': period_results[0]['class'],
            'predicted_decoration': decoration_results[0]['class'],
            'predicted_vessel_type': vessel_results[0]['class'],
            'period_confidence': period_results[0]['confidence'],
            'decoration_confidence': decoration_results[0]['confidence'],
            'vessel_confidence': vessel_results[0]['confidence']
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def explain_classification(image_data):
    """Generate Grad-CAM heatmap and explanations for classification"""
    if not load_ml_model():
        return {'error': 'ML model not available'}

    try:
        import torch
        import torch.nn.functional as F
        from PIL import Image
        import numpy as np
        import cv2

        # Decode image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        original_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_array = np.array(original_img)

        # Transform for model
        img_tensor = ML_TRANSFORM(original_img).unsqueeze(0)

        # Hook to capture gradients and activations
        gradients = []
        activations = []

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        def forward_hook(module, input, output):
            activations.append(output)

        # Register hooks on the last conv layer of ResNet50
        target_layer = ML_MODEL.backbone.layer4[-1]
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        # Forward pass
        ML_MODEL.eval()
        img_tensor.requires_grad_(True)
        period_out, decoration_out, vessel_out = ML_MODEL(img_tensor)

        # Get predictions
        period_pred = period_out.argmax(1).item()
        decoration_pred = decoration_out.argmax(1).item()
        vessel_pred = vessel_out.argmax(1).item()

        period_probs = torch.softmax(period_out, dim=1)
        decoration_probs = torch.softmax(decoration_out, dim=1)
        vessel_probs = torch.softmax(vessel_out, dim=1)

        # Compute Grad-CAM for decoration (most visual task)
        ML_MODEL.zero_grad()
        decoration_out[0, decoration_pred].backward(retain_graph=True)

        # Get gradients and activations
        grads = gradients[0].detach().cpu().numpy()[0]  # [C, H, W]
        acts = activations[0].detach().cpu().numpy()[0]  # [C, H, W]

        # Global average pooling of gradients
        weights = np.mean(grads, axis=(1, 2))  # [C]

        # Weighted combination of activation maps
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        # ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Create heatmap overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Blend with original
        overlay = (0.4 * heatmap + 0.6 * img_array).astype(np.uint8)

        # Convert to base64
        overlay_img = Image.fromarray(overlay)
        buffered = io.BytesIO()
        overlay_img.save(buffered, format="PNG")
        heatmap_b64 = base64.b64encode(buffered.getvalue()).decode()

        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()

        # Get class names
        period_name = ML_ENCODERS['period_classes'][period_pred]
        decoration_name = ML_ENCODERS['decoration_classes'][decoration_pred]
        vessel_name = ML_ENCODERS['vessel_classes'][vessel_pred]

        period_conf = float(period_probs[0, period_pred].detach()) * 100
        decoration_conf = float(decoration_probs[0, decoration_pred].detach()) * 100
        vessel_conf = float(vessel_probs[0, vessel_pred].detach()) * 100

        # Generate explanations
        explanations = generate_explanations(
            period_name, period_conf,
            decoration_name, decoration_conf,
            vessel_name, vessel_conf,
            cam
        )

        return {
            'success': True,
            'heatmap': f'data:image/png;base64,{heatmap_b64}',
            'predictions': {
                'period': {'class': period_name, 'confidence': period_conf},
                'decoration': {'class': decoration_name, 'confidence': decoration_conf},
                'vessel_type': {'class': vessel_name, 'confidence': vessel_conf}
            },
            'explanations': explanations
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def generate_explanations(period, period_conf, decoration, decoration_conf, vessel, vessel_conf, cam):
    """Generate human-readable explanations for predictions"""

    explanations = {
        'summary': '',
        'period_reason': '',
        'decoration_reason': '',
        'vessel_reason': '',
        'visual_focus': ''
    }

    # Period explanations
    period_reasons = {
        'Umm an-Nar': "Periodo Umm an-Nar (2700-2000 a.C.): caratterizzato da ceramica dipinta con motivi geometrici neri su fondo chiaro, tipica dell'Oman settentrionale.",
        'Wadi Suq': "Periodo Wadi Suq (2000-1600 a.C.): transizione con ceramica più semplice, spesso non decorata o con decorazioni incise.",
        'Late Bronze Age': "Tarda Età del Bronzo (1600-1250 a.C.): ceramica con influenze mesopotamiche, spesso con decorazioni a bande.",
        '': "Periodo non determinabile con certezza dalle caratteristiche visive."
    }

    # Decoration explanations
    decoration_reasons = {
        'painted': "Decorazione dipinta: il modello ha identificato pattern di pigmento (nero, rosso o bruno) applicati sulla superficie.",
        'plain': "Ceramica non decorata: superficie liscia senza motivi decorativi visibili.",
        'decorated': "Decorazione generica: presenza di elementi ornamentali non classificabili specificamente.",
        'incised': "Decorazione incisa: il modello ha rilevato linee incise o graffite sulla superficie.",
        'black-on-grey': "Black-on-grey: decorazione tipica con motivi neri su fondo grigio chiaro.",
        '': "Decorazione non determinabile."
    }

    # Vessel explanations
    vessel_reasons = {
        'plate': "Piatto: forma aperta e bassa con pareti poco profonde.",
        'bowl': "Ciotola: forma aperta con pareti arrotondate e profondità media.",
        'flask': "Fiasco/Fiaschetta: contenitore con collo stretto e corpo espanso.",
        'jar': "Giara: contenitore da stoccaggio con apertura larga.",
        'pot': "Pentola: recipiente profondo per cottura o conservazione.",
        'ceramic': "Frammento ceramico generico non classificabile.",
        'base': "Frammento di base del vaso.",
        'body sherd': "Frammento di parete del vaso.",
        'dish': "Piatto fondo o scodella.",
        'carinated bowl': "Ciotola carenata con spigolo distintivo.",
        'miniature vessel': "Vaso miniaturistico, possibile uso votivo.",
        'unknown': "Forma non identificabile."
    }

    # Build explanations
    explanations['period_reason'] = period_reasons.get(period, f"Periodo: {period}")
    explanations['decoration_reason'] = decoration_reasons.get(decoration, f"Decorazione: {decoration}")
    explanations['vessel_reason'] = vessel_reasons.get(vessel, f"Tipo: {vessel}")

    # Confidence-based summary
    avg_conf = (period_conf + decoration_conf + vessel_conf) / 3

    if avg_conf > 90:
        confidence_text = "Il modello è molto sicuro di questa classificazione."
    elif avg_conf > 70:
        confidence_text = "Il modello ha una buona confidenza, ma potrebbero esserci alternative."
    elif avg_conf > 50:
        confidence_text = "Classificazione incerta, verificare manualmente."
    else:
        confidence_text = "Bassa confidenza, il frammento potrebbe essere ambiguo."

    # Visual focus based on CAM
    cam_center = cam[cam.shape[0]//4:3*cam.shape[0]//4, cam.shape[1]//4:3*cam.shape[1]//4].mean()
    cam_edges = (cam[:cam.shape[0]//4].mean() + cam[3*cam.shape[0]//4:].mean()) / 2

    if cam_center > cam_edges * 1.3:
        focus_area = "Il modello si concentra principalmente sulla parte centrale del frammento, probabilmente sulla decorazione o forma del corpo."
    elif cam_edges > cam_center * 1.3:
        focus_area = "Il modello analizza principalmente i bordi e il profilo, utili per identificare la forma del vaso."
    else:
        focus_area = "Il modello considera uniformemente l'intera superficie del frammento."

    explanations['visual_focus'] = focus_area
    explanations['summary'] = f"{confidence_text} {focus_area}"

    return explanations


def explain_similarity(img1_path, img2_data, similarity_score):
    """Explain why two images are similar"""
    try:
        import torch
        from PIL import Image
        import numpy as np

        if not load_embeddings():
            return {'error': 'Embeddings not available'}

        # Load both images
        img1 = Image.open(img1_path).convert('RGB')

        if ',' in img2_data:
            img2_data = img2_data.split(',')[1]
        img2_bytes = base64.b64decode(img2_data)
        img2 = Image.open(io.BytesIO(img2_bytes)).convert('RGB')

        # Get features for comparison
        transform = ML_TRANSFORM

        with torch.no_grad():
            feat1 = FEATURE_EXTRACTOR(transform(img1).unsqueeze(0)).squeeze().numpy()
            feat2 = FEATURE_EXTRACTOR(transform(img2).unsqueeze(0)).squeeze().numpy()

        # Find which feature dimensions contribute most to similarity
        # Cosine similarity breakdown
        feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-8)
        feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-8)

        contributions = feat1_norm * feat2_norm
        top_features = np.argsort(contributions)[-10:]  # Top 10 contributing features

        # Map features to semantic meaning (simplified)
        # In a real scenario, you'd analyze what each feature dimension represents
        similarity_aspects = []

        if similarity_score > 0.8:
            similarity_aspects.append("Decorazione molto simile con pattern quasi identici")
            similarity_aspects.append("Stessa tecnica decorativa")
        elif similarity_score > 0.6:
            similarity_aspects.append("Pattern decorativi correlati")
            similarity_aspects.append("Stile artistico simile")
        elif similarity_score > 0.4:
            similarity_aspects.append("Alcune caratteristiche decorative in comune")
            similarity_aspects.append("Possibile stessa tradizione ceramica")
        else:
            similarity_aspects.append("Somiglianza limitata, principalmente nella texture")

        # Feature analysis
        high_freq_features = np.sum(np.abs(feat1 - feat2) < 0.1) / len(feat1)
        if high_freq_features > 0.7:
            similarity_aspects.append("Texture della superficie molto simile")
        if high_freq_features > 0.5:
            similarity_aspects.append("Composizione generale comparabile")

        return {
            'success': True,
            'similarity_score': similarity_score,
            'aspects': similarity_aspects,
            'explanation': f"Similarità del {similarity_score*100:.1f}%: " + "; ".join(similarity_aspects[:3])
        }

    except Exception as e:
        return {'error': str(e)}


def load_embeddings():
    """Load pre-computed image embeddings for similarity search."""
    global EMBEDDINGS, EMBEDDINGS_METADATA, FEATURE_EXTRACTOR

    if EMBEDDINGS is not None:
        return True

    embeddings_path = ML_MODEL_DIR / "image_embeddings.npz"
    metadata_path = ML_MODEL_DIR / "embeddings_metadata.json"
    feature_extractor_path = ML_MODEL_DIR / "feature_extractor.pt"

    if not embeddings_path.exists() or not metadata_path.exists():
        print("Embeddings not found. Run compute_embeddings.py first.")
        return False

    try:
        import torch
        import torch.nn as nn
        from torchvision import transforms, models

        # Load embeddings
        data = np.load(embeddings_path)
        EMBEDDINGS = data['embeddings']

        # Load metadata
        with open(metadata_path) as f:
            EMBEDDINGS_METADATA = json.load(f)

        # Create feature extractor for new images
        class FeatureExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                resnet = models.resnet18(weights=None)
                self.features = nn.Sequential(*list(resnet.children())[:-1])

            def forward(self, x):
                x = self.features(x)
                return x.view(x.size(0), -1)

        FEATURE_EXTRACTOR = FeatureExtractor()

        # Try to load cached weights first, then download if needed
        if feature_extractor_path.exists():
            print("   Loading cached feature extractor weights...")
            FEATURE_EXTRACTOR.load_state_dict(torch.load(feature_extractor_path, map_location='cpu', weights_only=True))
        else:
            print("   Downloading ResNet18 weights (first time only)...")
            try:
                resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                FEATURE_EXTRACTOR.features.load_state_dict(
                    nn.Sequential(*list(resnet.children())[:-1]).state_dict()
                )
                # Cache weights for future use
                torch.save(FEATURE_EXTRACTOR.state_dict(), feature_extractor_path)
                print(f"   Cached weights to {feature_extractor_path}")
            except Exception as download_err:
                print(f"   Warning: Could not download weights: {download_err}")
                print("   Similarity search will use random weights (less accurate)")

        FEATURE_EXTRACTOR.eval()

        print(f"Loaded {len(EMBEDDINGS)} image embeddings for similarity search")
        return True

    except Exception as e:
        print(f"Error loading embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_similar_images(image_data, top_k=20, threshold=0.5):
    """Find visually similar images using cosine similarity."""
    if not load_embeddings():
        return {'error': 'Embeddings not available'}

    try:
        import torch
        from torchvision import transforms
        from PIL import Image

        # Decode image
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img_tensor = transform(img).unsqueeze(0)

        # Extract embedding
        with torch.no_grad():
            query_embedding = FEATURE_EXTRACTOR(img_tensor).numpy().flatten()
            query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # Compute cosine similarities
        similarities = np.dot(EMBEDDINGS, query_embedding)

        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        similar_items = []
        for idx in top_indices:
            sim_score = float(similarities[idx])
            if sim_score >= threshold:
                item_meta = EMBEDDINGS_METADATA['items'][idx]
                similar_items.append({
                    'id': item_meta['id'],
                    'image_path': item_meta['image_path'],
                    'macro_period': item_meta['macro_period'],
                    'period': item_meta['period'],
                    'decoration': item_meta['decoration'],
                    'vessel_type': item_meta['vessel_type'],
                    'collection': item_meta['collection'],
                    'page_ref': item_meta['page_ref'],
                    'source_pdf': item_meta['source_pdf'],
                    'similarity': round(sim_score * 100, 1)
                })

        # Generate analysis
        analysis = generate_similarity_analysis(similar_items)

        return {
            'success': True,
            'similar_items': similar_items,
            'total_compared': len(EMBEDDINGS),
            'analysis': analysis,
            'statistics': compute_similarity_statistics(similar_items)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def generate_similarity_analysis(similar_items):
    """Generate a descriptive archaeological analysis based on similar items found."""
    if not similar_items:
        return {
            'text': "No similar ceramics were found in the database with sufficient visual similarity.",
            'period_suggestion': None,
            'references': []
        }

    # Analyze period distribution
    periods = {}
    decorations = {}
    collections = {}
    references = []

    for item in similar_items[:10]:  # Use top 10 for analysis
        if item['macro_period']:
            periods[item['macro_period']] = periods.get(item['macro_period'], 0) + 1
        if item['decoration']:
            decorations[item['decoration']] = decorations.get(item['decoration'], 0) + 1
        if item['collection']:
            collections[item['collection']] = collections.get(item['collection'], 0) + 1

        # Build references
        if item['page_ref'] and item['source_pdf']:
            ref = {
                'id': item['id'],
                'collection': item['collection'],
                'page_ref': item['page_ref'],
                'source_pdf': item['source_pdf']
            }
            if ref not in references:
                references.append(ref)

    # Determine most likely period
    suggested_period = max(periods.items(), key=lambda x: x[1])[0] if periods else None
    period_confidence = (periods.get(suggested_period, 0) / len(similar_items[:10]) * 100) if suggested_period else 0

    # Determine decoration type
    suggested_decoration = max(decorations.items(), key=lambda x: x[1])[0] if decorations else None

    # Build sites mentioned
    sites_by_collection = {
        'Degli_Espositi': 'Hili (UAE)',
        'Righetti': 'Hili 8 (UAE)',
        'Pellegrino': 'Masafi, Dibba, Tell Abraq (UAE/Oman)'
    }

    sites = [sites_by_collection.get(col, col) for col in collections.keys()]

    # Generate text
    text_parts = []

    if suggested_period:
        text_parts.append(f"Based on visual similarity analysis, this ceramic fragment most likely dates to the **{suggested_period}** period ({period_confidence:.0f}% confidence based on top matches).")

    if suggested_decoration:
        text_parts.append(f"The decoration style appears to be **{suggested_decoration}**.")

    if sites:
        text_parts.append(f"Similar ceramics have been found at archaeological sites including: **{', '.join(sites)}**.")

    if similar_items:
        top_match = similar_items[0]
        text_parts.append(f"The closest visual match ({top_match['similarity']}% similarity) is **{top_match['id']}** from the {top_match['collection']} collection.")

    if references:
        text_parts.append(f"For bibliographic references, see the linked PDF documents below.")

    return {
        'text': ' '.join(text_parts),
        'period_suggestion': suggested_period,
        'period_confidence': period_confidence,
        'decoration_suggestion': suggested_decoration,
        'sites': sites,
        'references': references[:5]  # Top 5 references
    }


def compute_similarity_statistics(similar_items):
    """Compute statistics for the similarity results."""
    if not similar_items:
        return {}

    periods = {}
    decorations = {}
    collections = {}
    similarities = []

    for item in similar_items:
        similarities.append(item['similarity'])
        if item['macro_period']:
            periods[item['macro_period']] = periods.get(item['macro_period'], 0) + 1
        if item['decoration']:
            decorations[item['decoration']] = decorations.get(item['decoration'], 0) + 1
        if item['collection']:
            collections[item['collection']] = collections.get(item['collection'], 0) + 1

    return {
        'period_distribution': periods,
        'decoration_distribution': decorations,
        'collection_distribution': collections,
        'similarity_range': {
            'min': min(similarities) if similarities else 0,
            'max': max(similarities) if similarities else 0,
            'avg': sum(similarities) / len(similarities) if similarities else 0
        }
    }


DEFAULT_CONFIG = {
    "collections": {
        "Schmidt_Bat": {"name": "Schmidt - Bat (Oman)", "pdf": "PDFs/Schmidt_Bat.pdf", "color": "#9C27B0"},
        "Degli_Espositi": {"name": "Degli Espositi - Tesi MDE", "pdf": "PDFs/2- Capp.4-5-6+bibliografia.pdf", "color": "#4472C4"},
        "Righetti": {"name": "Righetti - Hili 8 / Wadi Suq", "pdf": "PDFs/Righetti_Thèse_Volume_II.pdf", "color": "#ED7D31"},
        "Pellegrino": {"name": "Pellegrino - Masafi / Dibba / Tell Abraq", "pdf": "PDFs/2021-11_Pellegrino_cér.pdf", "color": "#70AD47"}
    },
    "metadata_file": "ceramica_metadata.csv"
}


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def get_db():
    """Get database connection"""
    if DATA_DIR:
        db_path = DB_FILE  # Already absolute path
    else:
        db_path = Path(__file__).parent / DB_FILE
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize SQLite database"""
    # Create data directory if using persistent storage
    if DATA_DIR and not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"Created data directory: {DATA_DIR}")

    conn = get_db()
    cursor = conn.cursor()

    # Create items table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS items (
            id TEXT PRIMARY KEY,
            type TEXT,
            period TEXT,
            figure_num TEXT,
            page_num TEXT,
            pottery_id TEXT,
            caption_text TEXT,
            position TEXT,
            rotation INTEGER DEFAULT 0,
            folder TEXT,
            image_path TEXT,
            page_ref TEXT,
            collection TEXT,
            source_pdf TEXT,
            decoration TEXT,
            part_type TEXT,
            vessel_type TEXT,
            macro_period TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create vocabulary tables for dynamic fields
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vocabulary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            field TEXT NOT NULL,
            value TEXT NOT NULL,
            count INTEGER DEFAULT 1,
            UNIQUE(field, value)
        )
    ''')

    # Create users table (for future expansion)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'viewer',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()


def run_auto_migrations():
    """Run database migrations automatically on startup"""
    print("   Checking for database migrations...")

    conn = get_db()
    cursor = conn.cursor()

    # Get existing columns in items table
    cursor.execute("PRAGMA table_info(items)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    # Define new columns to add (column_name, type, default)
    new_columns = [
        ("motivo_decorativo", "TEXT", None),
        ("sintassi_decorativa", "TEXT", None),
        ("scala_metrica", "TEXT", None),
        ("larghezza_cm", "REAL", None),
        ("altezza_cm", "REAL", None),
        ("calibration_data", "TEXT", None),  # JSON for measurement calibration
    ]

    added = 0
    for col_name, col_type, default in new_columns:
        if col_name not in existing_columns:
            try:
                if default is not None:
                    cursor.execute(f"ALTER TABLE items ADD COLUMN {col_name} {col_type} DEFAULT {default}")
                else:
                    cursor.execute(f"ALTER TABLE items ADD COLUMN {col_name} {col_type}")
                print(f"   + Added column: {col_name}")
                added += 1
            except Exception as e:
                print(f"   ! Error adding {col_name}: {e}")

    # Add vocabulary entries for new fields if not exist
    vocabulary_entries = [
        # Motivo decorativo
        ("motivo_decorativo", "wavy lines"), ("motivo_decorativo", "geometric patterns"),
        ("motivo_decorativo", "triangular"), ("motivo_decorativo", "crosshatched"),
        ("motivo_decorativo", "incised lines"), ("motivo_decorativo", "painted bands"),
        ("motivo_decorativo", "dotted"), ("motivo_decorativo", "zigzag"),
        ("motivo_decorativo", "chevron"), ("motivo_decorativo", "spiral"),
        ("motivo_decorativo", "hatched triangles"), ("motivo_decorativo", "pendant triangles"),
        ("motivo_decorativo", "horizontal bands"), ("motivo_decorativo", "vertical lines"),
        ("motivo_decorativo", "net pattern"),
        # Sintassi decorativa
        ("sintassi_decorativa", "rim band"), ("sintassi_decorativa", "shoulder decoration"),
        ("sintassi_decorativa", "body decoration"), ("sintassi_decorativa", "base decoration"),
        ("sintassi_decorativa", "full coverage"), ("sintassi_decorativa", "register division"),
        ("sintassi_decorativa", "metope arrangement"), ("sintassi_decorativa", "frieze pattern"),
        ("sintassi_decorativa", "random distribution"), ("sintassi_decorativa", "symmetrical"),
        ("sintassi_decorativa", "asymmetrical"),
        # Scala metrica
        ("scala_metrica", "1:1"), ("scala_metrica", "1:2"), ("scala_metrica", "1:3"),
        ("scala_metrica", "1:4"), ("scala_metrica", "1:5"), ("scala_metrica", "2:3"),
    ]

    for field, value in vocabulary_entries:
        cursor.execute('''
            INSERT OR IGNORE INTO vocabulary (field, value, count)
            VALUES (?, ?, 0)
        ''', (field, value))

    conn.commit()
    conn.close()

    if added > 0:
        print(f"   Migration complete: {added} columns added")
    else:
        print("   Database schema is up to date")


def sync_bundled_data():
    """Sync data from bundled database to persistent database on Railway.
    This ensures new collections added to the app are copied to the volume.
    """
    if not DATA_DIR:
        return  # Not on Railway with persistent storage

    bundled_db = Path(__file__).parent / "ceramica.db"
    if not bundled_db.exists():
        print("   No bundled database found, skipping sync")
        return

    print("   Checking for missing collections in persistent database...")

    # Connect to persistent database
    persistent_conn = get_db()
    persistent_cursor = persistent_conn.cursor()

    # Get collections in persistent DB
    persistent_cursor.execute("SELECT DISTINCT collection FROM items")
    persistent_collections = {row[0] for row in persistent_cursor.fetchall()}

    # Connect to bundled database
    bundled_conn = sqlite3.connect(str(bundled_db))
    bundled_conn.row_factory = sqlite3.Row
    bundled_cursor = bundled_conn.cursor()

    # Get collections in bundled DB
    bundled_cursor.execute("SELECT DISTINCT collection FROM items")
    bundled_collections = {row[0] for row in bundled_cursor.fetchall()}

    # Find missing collections
    missing_collections = bundled_collections - persistent_collections

    if not missing_collections:
        print("   All collections are synced")
        bundled_conn.close()
        persistent_conn.close()
        return

    print(f"   Found missing collections: {missing_collections}")

    # Copy missing collections
    for collection in missing_collections:
        print(f"   + Syncing collection: {collection}")

        # Get all items from bundled DB for this collection
        bundled_cursor.execute("SELECT * FROM items WHERE collection = ?", (collection,))
        items = bundled_cursor.fetchall()

        if not items:
            continue

        # Get column names
        columns = [description[0] for description in bundled_cursor.description]

        # Insert into persistent DB
        placeholders = ','.join(['?' for _ in columns])
        columns_str = ','.join(columns)

        for item in items:
            try:
                persistent_cursor.execute(
                    f"INSERT OR REPLACE INTO items ({columns_str}) VALUES ({placeholders})",
                    tuple(item)
                )
            except Exception as e:
                print(f"     ! Error inserting {item['id']}: {e}")

        print(f"     Synced {len(items)} items from {collection}")

    persistent_conn.commit()
    bundled_conn.close()
    persistent_conn.close()
    print("   Database sync complete")


def safe_int(value, default=0):
    """Safely convert value to int"""
    if value is None or value == '':
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def migrate_csv_to_db():
    """Migrate data from CSV to SQLite"""
    csv_path = Path(__file__).parent / CSV_FILE
    if not csv_path.exists():
        return False

    conn = get_db()
    cursor = conn.cursor()

    # Check if migration already done
    cursor.execute("SELECT COUNT(*) FROM items")
    if cursor.fetchone()[0] > 0:
        conn.close()
        return True

    # Read CSV
    df = pd.read_csv(csv_path)
    df = df.fillna('')

    # Add macro_period
    df['macro_period'] = df['period'].apply(get_macro_period)

    # Insert items
    for _, row in df.iterrows():
        cursor.execute('''
            INSERT OR REPLACE INTO items
            (id, type, period, figure_num, page_num, pottery_id, caption_text,
             position, rotation, folder, image_path, page_ref, collection,
             source_pdf, decoration, part_type, vessel_type, macro_period)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row.get('id', ''), row.get('type', ''), row.get('period', ''),
            row.get('figure_num', ''), str(row.get('page_num', '')), row.get('pottery_id', ''),
            row.get('caption_text', ''), row.get('position', ''), safe_int(row.get('rotation', 0)),
            row.get('folder', ''), row.get('image_path', ''), row.get('page_ref', ''),
            row.get('collection', ''), row.get('source_pdf', ''), row.get('decoration', ''),
            row.get('part_type', ''), row.get('vessel_type', ''), row.get('macro_period', '')
        ))

    # Build vocabulary from existing data
    for field in ['decoration', 'vessel_type', 'part_type', 'period']:
        values = df[field].unique()
        for val in values:
            if val:
                cursor.execute('''
                    INSERT OR IGNORE INTO vocabulary (field, value, count)
                    VALUES (?, ?, 1)
                ''', (field, str(val)))

    conn.commit()
    conn.close()
    print(f"Migrated {len(df)} items to SQLite database")
    return True


def get_macro_period(period_str):
    """Determine macro-period from period string"""
    if not period_str:
        return ""
    period_lower = str(period_str).lower()
    for macro, keywords in MACRO_PERIODS.items():
        for keyword in keywords:
            if keyword in period_lower:
                return macro
    return ""


def get_all_items():
    """Get all items from database"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM items ORDER BY collection, id")
    items = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return items


def get_item(item_id):
    """Get single item by ID"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM items WHERE id = ?", (item_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def update_item(item_id, fields):
    """Update item fields"""
    conn = get_db()
    cursor = conn.cursor()

    # Update macro_period if period changed
    if 'period' in fields:
        fields['macro_period'] = get_macro_period(fields['period'])

    fields['updated_at'] = datetime.now().isoformat()

    # Build update query
    set_clause = ', '.join([f"{k} = ?" for k in fields.keys()])
    values = list(fields.values()) + [item_id]

    cursor.execute(f"UPDATE items SET {set_clause} WHERE id = ?", values)

    # Update vocabulary
    for field in ['decoration', 'vessel_type', 'part_type', 'period']:
        if field in fields and fields[field]:
            cursor.execute('''
                INSERT INTO vocabulary (field, value, count) VALUES (?, ?, 1)
                ON CONFLICT(field, value) DO UPDATE SET count = count + 1
            ''', (field, fields[field]))

    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0


def delete_item(item_id):
    """Delete item from database"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM items WHERE id = ?", (item_id,))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0


def get_vocabulary(field=None):
    """Get vocabulary for autocomplete"""
    conn = get_db()
    cursor = conn.cursor()

    if field:
        cursor.execute("SELECT field, value, count FROM vocabulary WHERE field = ? ORDER BY count DESC", (field,))
    else:
        cursor.execute("SELECT field, value, count FROM vocabulary ORDER BY field, count DESC")

    vocab = {}
    for row in cursor.fetchall():
        f = row['field']
        if f not in vocab:
            vocab[f] = []
        vocab[f].append({'value': row['value'], 'count': row['count']})

    conn.close()
    return vocab


def get_unique_periods():
    """Get all unique periods"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT period FROM items WHERE period != '' ORDER BY period")
    periods = [row['period'] for row in cursor.fetchall()]
    conn.close()
    return periods


def get_statistics():
    """Get database statistics"""
    conn = get_db()
    cursor = conn.cursor()

    stats = {}
    cursor.execute("SELECT COUNT(*) as total FROM items")
    stats['total_items'] = cursor.fetchone()['total']

    cursor.execute("SELECT collection, COUNT(*) as count FROM items GROUP BY collection")
    stats['by_collection'] = {row['collection']: row['count'] for row in cursor.fetchall()}

    cursor.execute("SELECT macro_period, COUNT(*) as count FROM items WHERE macro_period != '' GROUP BY macro_period")
    stats['by_macro_period'] = {row['macro_period']: row['count'] for row in cursor.fetchall()}

    conn.close()
    return stats


def get_decoration_catalog():
    """Get decoration classification catalog (Schmidt Bat)"""
    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT code, category, name_de, name_it, name_en,
                   description, is_combination, base_codes, period, source,
                   thumbnail_path, pdf_page, pdf_reference, period_diagnostic
            FROM decoration_catalog
            ORDER BY code
        """)

        catalog = []
        for row in cursor.fetchall():
            catalog.append({
                'code': row['code'],
                'category': row['category'],
                'name_de': row['name_de'],
                'name_it': row['name_it'],
                'name_en': row['name_en'],
                'description': row['description'],
                'is_combination': bool(row['is_combination']),
                'base_codes': row['base_codes'].split(',') if row['base_codes'] else [],
                'period': row['period'],
                'source': row['source'],
                'thumbnail': row['thumbnail_path'],
                'pdf_page': row['pdf_page'],
                'pdf_reference': row['pdf_reference'],
                'period_diagnostic': row['period_diagnostic']
            })

        conn.close()
        return catalog
    except:
        conn.close()
        return []


# ============================================================================
# AUTHENTICATION FUNCTIONS
# ============================================================================

def hash_password(password):
    """Hash password with SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_credentials(password):
    """Verify password and return role"""
    pwd_hash = hash_password(password)
    if pwd_hash == ADMIN_HASH:
        return 'admin'
    elif pwd_hash == VIEWER_HASH:
        return 'viewer'
    return None


def create_session(role):
    """Create session token"""
    token = secrets.token_hex(32)
    SESSIONS[token] = {'role': role, 'created': datetime.now().isoformat()}
    return token


def get_session(cookie_header):
    """Get session from cookie"""
    if not cookie_header:
        return None
    cookies = http.cookies.SimpleCookie()
    try:
        cookies.load(cookie_header)
        if 'session' in cookies:
            token = cookies['session'].value
            return SESSIONS.get(token)
    except:
        pass
    return None


def is_admin(cookie_header):
    """Check if session is admin"""
    session = get_session(cookie_header)
    return session and session.get('role') == 'admin'


def is_authenticated(cookie_header):
    """Check if session is valid (admin or viewer)"""
    return get_session(cookie_header) is not None


# ============================================================================
# IMAGE FUNCTIONS
# ============================================================================

def rotate_image(image_path, degrees):
    """Rotate image by specified degrees"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, degrees, 1.0)

        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(img, M, (new_w, new_h),
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(255, 255, 255))
        cv2.imwrite(image_path, rotated)
        return True
    except Exception as e:
        print(f"Rotation error: {e}")
        return False


def flip_image(image_path, direction):
    """Flip image horizontally or vertically
    direction: 'horizontal' (mirror) or 'vertical'
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False

        if direction == 'horizontal':
            flipped = cv2.flip(img, 1)  # 1 = horizontal flip (mirror)
        elif direction == 'vertical':
            flipped = cv2.flip(img, 0)  # 0 = vertical flip
        else:
            return False

        cv2.imwrite(image_path, flipped)
        return True
    except Exception as e:
        print(f"Flip error: {e}")
        return False


def delete_image_file(image_path):
    """Delete image file from filesystem"""
    base_path = Path(__file__).parent
    full_path = base_path / image_path
    if full_path.exists():
        os.remove(full_path)
        return True
    return False


# ============================================================================
# CONFIG FUNCTIONS
# ============================================================================

def load_config():
    """Load configuration"""
    config_path = Path(__file__).parent / CONFIG_FILE
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return DEFAULT_CONFIG


# ============================================================================
# HTTP HANDLER
# ============================================================================

class ViewerHandler(SimpleHTTPRequestHandler):

    def send_json(self, data, status=200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str, ensure_ascii=False).encode('utf-8'))

    def send_html(self, content, status=200):
        """Send HTML response"""
        self.send_response(status)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))

    def get_role(self):
        """Get current user role"""
        session = get_session(self.headers.get('Cookie'))
        return session.get('role') if session else None

    def require_auth(self):
        """Check authentication, return role or None"""
        role = self.get_role()
        if not role:
            self.send_response(302)
            self.send_header('Location', '/')
            self.end_headers()
            return None
        return role

    def require_admin(self):
        """Check admin authentication"""
        role = self.get_role()
        if role != 'admin':
            self.send_json({'error': 'Admin access required'}, 403)
            return False
        return True

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)

        # ===== PUBLIC PAGES =====

        # Welcome/Login page
        if parsed.path == '/' or parsed.path == '/login':
            self.send_html(WELCOME_PAGE)
            return

        # ===== PUBLIC API (for ML) =====

        # Get all items (public for ML)
        if parsed.path == '/api/v1/items':
            items = get_all_items()
            self.send_json({'items': items, 'total': len(items)})
            return

        # Get single item (public for ML)
        if parsed.path.startswith('/api/v1/items/'):
            item_id = parsed.path.replace('/api/v1/items/', '')
            item = get_item(urllib.parse.unquote(item_id))
            if item:
                self.send_json(item)
            else:
                self.send_json({'error': 'Item not found'}, 404)
            return

        # Get vocabulary (public for ML)
        if parsed.path == '/api/v1/vocabulary':
            field = query.get('field', [None])[0]
            vocab = get_vocabulary(field)
            self.send_json(vocab)
            return

        # Get unique periods (public)
        if parsed.path == '/api/v1/periods':
            periods = get_unique_periods()
            self.send_json({'periods': periods})
            return

        # Get statistics (public)
        if parsed.path == '/api/v1/stats':
            stats = get_statistics()
            self.send_json(stats)
            return

        # Get decoration catalog (public - Schmidt classification)
        if parsed.path == '/api/v1/decoration-catalog':
            catalog = get_decoration_catalog()
            self.send_json({'catalog': catalog, 'count': len(catalog)})
            return

        # ===== PROTECTED PAGES =====

        # Viewer page
        if parsed.path == '/viewer':
            role = self.require_auth()
            if not role:
                return
            self.send_html(get_viewer_html(role))
            return

        # ===== PROTECTED API =====

        # Get config (authenticated)
        if parsed.path == '/api/config':
            config = load_config()
            config['macro_periods'] = list(MACRO_PERIODS.keys())
            config['user_role'] = self.get_role()
            self.send_json(config)
            return

        # Get data (authenticated)
        if parsed.path == '/api/data':
            items = get_all_items()
            self.send_json(items)
            return

        # Get PDF URL for cross-platform viewing
        if parsed.path.startswith('/api/pdf-url'):
            page = query.get('page', ['1'])[0]
            collection = query.get('collection', [''])[0]

            # Extract page number from various formats
            page_match = re.search(r'p+\.?\s*(\d+)', page)
            page_num = page_match.group(1) if page_match else '1'

            config = load_config()
            pdf_path = config.get('collections', {}).get(collection, {}).get('pdf', '')

            if pdf_path:
                # Check if PDF exists
                base_path = Path(__file__).parent
                full_pdf_path = base_path / pdf_path
                if full_pdf_path.exists():
                    # Return URL with page fragment for browser PDF viewer
                    pdf_url = f'/{pdf_path}#page={page_num}'
                    self.send_json({'success': True, 'url': pdf_url, 'page': page_num})
                else:
                    self.send_json({'error': 'PDF file not found', 'path': pdf_path})
            else:
                self.send_json({'error': 'No PDF configured for this collection'})
            return

        # Legacy endpoint for local macOS use
        if parsed.path.startswith('/api/open-pdf'):
            page = query.get('page', ['1'])[0]
            collection = query.get('collection', [''])[0]
            direct_pdf = query.get('pdf', [''])[0]  # Direct PDF path support

            page_match = re.search(r'p+\.?\s*(\d+)', page)
            page_num = page_match.group(1) if page_match else page if page.isdigit() else '1'

            if sys.platform == 'darwin':
                # Check for direct PDF path first
                if direct_pdf and os.path.exists(direct_pdf):
                    full_pdf_path = direct_pdf
                else:
                    config = load_config()
                    pdf_path = config.get('collections', {}).get(collection, {}).get('pdf', '')
                    if pdf_path:
                        base_path = Path(__file__).parent
                        full_pdf_path = str(base_path / pdf_path)
                    else:
                        full_pdf_path = None

                if full_pdf_path:
                    script = f'''
                    do shell script "open -b com.apple.Preview " & quoted form of "{full_pdf_path}"
                    delay 2.5
                    tell application "Preview" to activate
                    delay 1
                    tell application "System Events"
                        tell process "Preview"
                            click menu item "Vai alla pagina…" of menu "Vai" of menu bar 1
                            delay 0.5
                            keystroke "{page_num}"
                            delay 0.3
                            key code 36
                        end tell
                    end tell
                    '''
                    subprocess.run(['osascript', '-e', script], check=False)
                    self.send_json({'success': True, 'page': page_num, 'method': 'local'})
                else:
                    self.send_json({'error': 'PDF not found'})
            else:
                # On non-macOS, redirect to browser-based viewing
                config = load_config()
                pdf_path = config.get('collections', {}).get(collection, {}).get('pdf', '')
                if pdf_path:
                    pdf_url = f'/{pdf_path}#page={page_num}'
                    self.send_json({'success': True, 'url': pdf_url, 'page': page_num, 'method': 'browser'})
                else:
                    self.send_json({'error': 'PDF not found'})
            return

        # Extract scale from PDF
        if parsed.path.startswith('/api/extract-scale'):
            collection = query.get('collection', [''])[0]
            page = query.get('page', [None])[0]

            config = load_config()
            pdf_path = config.get('collections', {}).get(collection, {}).get('pdf', '')

            if pdf_path:
                base_path = Path(__file__).parent
                full_pdf_path = str(base_path / pdf_path)

                page_num = None
                if page:
                    page_match = re.search(r'(\d+)', page)
                    if page_match:
                        page_num = int(page_match.group(1))

                result = extract_scale_from_pdf(full_pdf_path, page_num)
                self.send_json(result)
            else:
                self.send_json({'error': 'No PDF configured for this collection'})
            return

        # Serve static files with caching
        # Add cache headers for images and PDFs
        if parsed.path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
            try:
                return self.serve_static_with_cache(parsed.path, 'image', max_age=86400)  # 1 day
            except Exception as e:
                print(f"Error serving image {parsed.path}: {e}")
                pass
        elif parsed.path.endswith('.pdf'):
            try:
                return self.serve_static_with_cache(parsed.path, 'pdf', max_age=604800)  # 1 week
            except Exception as e:
                print(f"Error serving PDF {parsed.path}: {e}")
                # Check if it's an LFS pointer file issue
                file_path = Path(__file__).parent / parsed.path.lstrip('/')
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        header = f.read(100)
                        if b'version https://git-lfs.github.com' in header:
                            self.send_error(503, "PDF files not available - Git LFS not configured on server")
                            return
                pass

        return SimpleHTTPRequestHandler.do_GET(self)

    def serve_static_with_cache(self, path, file_type, max_age=86400):
        """Serve static files with cache headers"""
        # URL decode path (handle %20 for spaces, etc.)
        decoded_path = urllib.parse.unquote(path.lstrip('/'))
        file_path = Path(__file__).parent / decoded_path
        if not file_path.exists():
            print(f"File not found: {file_path}")
            self.send_error(404, f"File not found: {decoded_path}")
            return

        # Determine content type
        content_types = {
            'image': 'image/png',
            'pdf': 'application/pdf'
        }
        if path.endswith('.jpg') or path.endswith('.jpeg'):
            content_types['image'] = 'image/jpeg'
        elif path.endswith('.gif'):
            content_types['image'] = 'image/gif'
        elif path.endswith('.webp'):
            content_types['image'] = 'image/webp'

        self.send_response(200)
        self.send_header('Content-Type', content_types.get(file_type, 'application/octet-stream'))
        self.send_header('Cache-Control', f'public, max-age={max_age}')
        self.send_header('Access-Control-Allow-Origin', '*')

        # Stream file
        with open(file_path, 'rb') as f:
            content = f.read()
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = json.loads(self.rfile.read(content_length).decode('utf-8')) if content_length > 0 else {}

        # Login
        if parsed.path == '/api/login':
            password = post_data.get('password', '')
            role = verify_credentials(password)

            if role:
                token = create_session(role)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Set-Cookie', f'session={token}; Path=/; HttpOnly; SameSite=Strict')
                self.end_headers()
                self.wfile.write(json.dumps({'success': True, 'role': role}).encode())
            else:
                self.send_json({'success': False, 'error': 'Invalid credentials'})
            return

        # Logout
        if parsed.path == '/api/logout':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Set-Cookie', 'session=; Path=/; Max-Age=0')
            self.end_headers()
            self.wfile.write(json.dumps({'success': True}).encode())
            return

        # ===== ADMIN ONLY ENDPOINTS =====

        # Update item
        if parsed.path == '/api/update-item':
            if not self.require_admin():
                return

            item_id = post_data.get('id')
            fields = post_data.get('fields', {})

            if update_item(item_id, fields):
                self.send_json({'success': True})
            else:
                self.send_json({'success': False, 'error': 'Update failed'})
            return

        # Batch update
        if parsed.path == '/api/update-batch':
            if not self.require_admin():
                return

            ids = post_data.get('ids', [])
            fields = post_data.get('fields', {})

            updated = 0
            for item_id in ids:
                if update_item(item_id, fields):
                    updated += 1

            self.send_json({'success': True, 'updated': updated})
            return

        # Rotate image
        if parsed.path == '/api/rotate-image':
            if not self.require_admin():
                return

            image_path = post_data.get('path', '')
            degrees = post_data.get('degrees', 90)

            base_path = Path(__file__).parent
            full_path = str(base_path / image_path)

            if rotate_image(full_path, degrees):
                self.send_json({'success': True})
            else:
                self.send_json({'success': False, 'error': 'Rotation failed'})
            return

        # Flip image (horizontal/vertical)
        if parsed.path == '/api/flip-image':
            if not self.require_admin():
                return

            image_path = post_data.get('path', '')
            direction = post_data.get('direction', 'horizontal')

            base_path = Path(__file__).parent
            full_path = str(base_path / image_path)

            if flip_image(full_path, direction):
                self.send_json({'success': True})
            else:
                self.send_json({'success': False, 'error': 'Flip failed'})
            return

        # Add vocabulary term
        if parsed.path == '/api/vocabulary':
            if not self.require_admin():
                return

            field = post_data.get('field')
            value = post_data.get('value')

            if field and value:
                conn = get_db()
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO vocabulary (field, value, count) VALUES (?, ?, 1)
                    ON CONFLICT(field, value) DO UPDATE SET count = count + 1
                ''', (field, value))
                conn.commit()
                conn.close()
                self.send_json({'success': True})
            else:
                self.send_json({'success': False, 'error': 'Field and value required'})
            return

        # ML Preprocess endpoint - convert real photo to drawing
        if parsed.path == '/api/ml/preprocess':
            image_data = post_data.get('image')
            if not image_data:
                self.send_json({'error': 'No image provided'}, 400)
                return

            # Get custom parameters if provided
            params = post_data.get('params', {})
            threshold = params.get('threshold', 18)
            min_area = params.get('minArea', 0.05)
            blur = params.get('blur', 9)

            result = photo_to_drawing(image_data, threshold=threshold, min_area_pct=min_area, blur_size=blur)
            self.send_json(result)
            return

        # ML Combine Drawing endpoint - combine user drawing with auto contour
        if parsed.path == '/api/ml/combine-drawing':
            original_image = post_data.get('original_image')
            drawing_image = post_data.get('drawing')
            preprocessed_image = post_data.get('preprocessed_image')  # Optional

            if not original_image:
                self.send_json({'error': 'No original image provided'}, 400)
                return

            if not drawing_image:
                self.send_json({'error': 'No drawing provided'}, 400)
                return

            result = combine_drawing_with_contour(original_image, drawing_image, preprocessed_image)
            self.send_json(result)
            return

        # ML Classification endpoint (public)
        if parsed.path == '/api/ml/classify':
            image_data = post_data.get('image')
            preprocess = post_data.get('preprocess', False)

            if not image_data:
                self.send_json({'error': 'No image provided'}, 400)
                return

            # Apply preprocessing if real photo mode enabled
            if preprocess:
                preprocess_result = photo_to_drawing(image_data)
                if preprocess_result.get('success'):
                    image_data = preprocess_result['processed_image']

            result = classify_image(image_data)
            self.send_json(result)
            return

        # ML Explain Classification endpoint (with Grad-CAM)
        if parsed.path == '/api/ml/explain':
            image_data = post_data.get('image')
            preprocess = post_data.get('preprocess', False)

            if not image_data:
                self.send_json({'error': 'No image provided'}, 400)
                return

            # Apply preprocessing if real photo mode enabled
            if preprocess:
                preprocess_result = photo_to_drawing(image_data)
                if preprocess_result.get('success'):
                    image_data = preprocess_result['processed_image']

            result = explain_classification(image_data)
            self.send_json(result)
            return

        # Image Similarity Search endpoint
        if parsed.path == '/api/ml/similar':
            try:
                image_data = post_data.get('image')
                top_k = post_data.get('top_k', 20)
                threshold = post_data.get('threshold', 0.3)
                preprocess = post_data.get('preprocess', False)

                if not image_data:
                    self.send_json({'error': 'No image provided'}, 400)
                    return

                # Apply preprocessing if real photo mode enabled
                if preprocess:
                    preprocess_result = photo_to_drawing(image_data)
                    if preprocess_result.get('success'):
                        image_data = preprocess_result['processed_image']

                result = find_similar_images(image_data, top_k=top_k, threshold=threshold)
                self.send_json(result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_json({'error': f'Similarity search failed: {str(e)}', 'similar_items': []}, 500)
            return

        # Get all images for carousel animation
        if parsed.path == '/api/ml/all-images':
            try:
                if EMBEDDINGS_METADATA:
                    images = [{'id': item['id'], 'image_path': item['image_path']}
                              for item in EMBEDDINGS_METADATA['items']]
                else:
                    # Load from database
                    conn = get_db()
                    cursor = conn.cursor()
                    cursor.execute("SELECT id, image_path FROM items WHERE image_path != ''")
                    images = [{'id': row[0], 'image_path': row[1]} for row in cursor.fetchall()]
                    conn.close()
                self.send_json({'images': images, 'total': len(images)})
            except Exception as e:
                self.send_json({'error': str(e)}, 500)
            return

        # Check ML model status
        if parsed.path == '/api/ml/status':
            model_exists = ML_MODEL_PATH.exists()
            encoders_exist = ML_ENCODERS_PATH.exists()
            self.send_json({
                'available': model_exists and encoders_exist,
                'model_loaded': ML_MODEL is not None,
                'model_path': str(ML_MODEL_PATH),
                'encoders_path': str(ML_ENCODERS_PATH)
            })
            return

        # 3D Reconstruction endpoint
        if parsed.path == '/api/3d/reconstruct':
            image_path = post_data.get('image_path', '')
            with_decoration = post_data.get('with_decoration', True)
            use_ai = post_data.get('use_ai', False)
            api_key = post_data.get('api_key', '')
            print(f"[3D] image={image_path}, use_ai={use_ai}, has_api_key={bool(api_key)}", flush=True)

            if not image_path:
                self.send_json({'error': 'No image path provided'}, 400)
                return

            base_path = Path(__file__).parent
            full_path = str(base_path / image_path)

            if not os.path.exists(full_path):
                self.send_json({'error': 'Image not found'}, 404)
                return

            try:
                from vessel_3d_reconstruction import reconstruct_vessel
                result = reconstruct_vessel(
                    full_path,
                    debug=False,
                    with_decoration=with_decoration,
                    use_ai=use_ai,
                    api_key=api_key if api_key else None
                )

                if result and result.get('glb_path'):
                    # Read GLB file and encode as base64
                    with open(result['glb_path'], 'rb') as f:
                        glb_data = base64.b64encode(f.read()).decode('utf-8')

                    response = {
                        'success': True,
                        'glb_data': glb_data,
                        'vertices': len(result['mesh']['vertices']),
                        'faces': len(result['mesh']['faces']),
                        'profile_points': result['mesh']['profile_points'],
                        'has_thickness': result['mesh'].get('has_thickness', False),
                        'wall_thickness': result['profile'].get('avg_thickness', 0)
                    }

                    # Include textured GLB if available
                    if result.get('glb_textured_path') and os.path.exists(result['glb_textured_path']):
                        with open(result['glb_textured_path'], 'rb') as f:
                            response['glb_textured_data'] = base64.b64encode(f.read()).decode('utf-8')
                        response['has_decoration'] = result.get('decoration', {}).get('has_decoration', False)

                    # Include AI-generated texture GLB if available
                    if result.get('glb_ai_path') and os.path.exists(result['glb_ai_path']):
                        with open(result['glb_ai_path'], 'rb') as f:
                            response['glb_ai_data'] = base64.b64encode(f.read()).decode('utf-8')
                        response['ai_analysis'] = result.get('ai_analysis')
                        response['has_ai_decoration'] = True

                    self.send_json(response)
                else:
                    self.send_json({'error': 'Failed to extract profile from image'}, 400)
            except Exception as e:
                self.send_json({'error': f'3D reconstruction failed: {str(e)}'}, 500)
            return

        self.send_json({'error': 'Not found'}, 404)

    def do_DELETE(self):
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)

        if not self.require_admin():
            return

        # Delete single item
        if parsed.path == '/api/delete-image':
            image_path = query.get('path', [''])[0]
            item_id = query.get('id', [''])[0]

            # Delete file
            delete_image_file(image_path)

            # Delete from database
            if delete_item(item_id):
                self.send_json({'success': True})
            else:
                self.send_json({'success': False, 'error': 'Delete failed'})
            return

        self.send_json({'error': 'Not found'}, 404)

    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()


# ============================================================================
# HTML TEMPLATES
# ============================================================================

WELCOME_PAGE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CeramicaDatabase - Archaeological Ceramic Collections</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        .hero {
            text-align: center;
            padding: 60px 20px;
        }
        .logo { font-size: 5em; margin-bottom: 20px; }
        h1 { color: #4fc3f7; font-size: 2.5em; margin-bottom: 15px; }
        .tagline { color: #888; font-size: 1.2em; margin-bottom: 40px; }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            margin-bottom: 50px;
        }
        .stat-card {
            background: rgba(255,255,255,0.05);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .stat-number { font-size: 2.5em; font-weight: bold; color: #4fc3f7; }
        .stat-label { color: #888; margin-top: 5px; }

        .section { margin-bottom: 50px; }
        .section h2 { color: #4fc3f7; margin-bottom: 20px; font-size: 1.5em; }
        .section p { line-height: 1.8; color: #bbb; margin-bottom: 15px; }

        .collections {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        .collection-card {
            background: rgba(255,255,255,0.05);
            padding: 25px;
            border-radius: 12px;
            border-left: 4px solid var(--color);
        }
        .collection-card h3 { color: var(--color); margin-bottom: 10px; }
        .collection-card p { font-size: 0.9em; color: #888; }

        .periods-list {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-top: 20px;
        }
        .period-badge {
            background: rgba(255,152,0,0.2);
            color: #ff9800;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.85em;
        }

        .api-section {
            background: rgba(0,0,0,0.3);
            padding: 30px;
            border-radius: 12px;
            margin-top: 30px;
        }
        .api-endpoint {
            background: rgba(255,255,255,0.05);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            font-family: monospace;
        }
        .api-endpoint .method { color: #4caf50; font-weight: bold; }
        .api-endpoint .path { color: #4fc3f7; }
        .api-endpoint .desc { color: #888; font-size: 0.85em; margin-top: 5px; font-family: sans-serif; }

        .login-section {
            background: rgba(255,255,255,0.05);
            padding: 40px;
            border-radius: 20px;
            max-width: 400px;
            margin: 50px auto;
            text-align: center;
        }
        .login-section h3 { color: #4fc3f7; margin-bottom: 25px; }
        .login-form { display: flex; flex-direction: column; gap: 15px; }
        .login-form input {
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            color: #fff;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .login-form input:focus { outline: none; border-color: #4fc3f7; }
        .login-form button {
            background: linear-gradient(135deg, #4fc3f7, #29b6f6);
            color: #000;
            border: none;
            padding: 15px;
            border-radius: 10px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .login-form button:hover { transform: translateY(-2px); }
        .error { color: #ff5252; display: none; margin-top: 10px; }
        .error.show { display: block; }
        .role-info { font-size: 0.8em; color: #666; margin-top: 20px; }

        .footer {
            text-align: center;
            padding: 40px;
            color: #666;
            border-top: 1px solid rgba(255,255,255,0.1);
            margin-top: 50px;
        }
        .footer a { color: #4fc3f7; text-decoration: none; }
    </style>
</head>
<body>
    <div class="container">
        <div class="hero">
            <div class="logo">&#127994;</div>
            <h1>CeramicaDatabase</h1>
            <p class="tagline">Unified Archaeological Ceramic Collections Viewer</p>
        </div>

        <div class="stats-grid" id="stats">
            <div class="stat-card"><div class="stat-number">797</div><div class="stat-label">Artifacts</div></div>
            <div class="stat-card"><div class="stat-number">3</div><div class="stat-label">Collections</div></div>
            <div class="stat-card"><div class="stat-number">3</div><div class="stat-label">Chronological Periods</div></div>
        </div>

        <div class="section">
            <h2>About This Project</h2>
            <p>CeramicaDatabase is a comprehensive digital archive consolidating ceramic artifact collections from three major archaeological research projects focused on Bronze Age and Iron Age sites in the Arabian Peninsula.</p>
            <p>The database provides high-resolution images of ceramic fragments and complete vessels, cross-referenced with academic publications, organized by chronological period, and enriched with typological metadata including vessel type, decoration patterns, and morphological characteristics.</p>
        </div>

        <div class="section">
            <h2>Collections</h2>
            <div class="collections">
                <div class="collection-card" style="--color: #4472C4">
                    <h3>Degli Espositi Collection</h3>
                    <p><strong>232 artifacts</strong></p>
                    <p>Ceramics from Sequence T7 - ST1, documenting Umm an-Nar and Wadi Suq periods.</p>
                </div>
                <div class="collection-card" style="--color: #ED7D31">
                    <h3>Righetti Collection</h3>
                    <p><strong>210 artifacts</strong></p>
                    <p>Ceramics from Hili 8, representing the Umm an-Nar period (3rd millennium BCE).</p>
                </div>
                <div class="collection-card" style="--color: #70AD47">
                    <h3>Pellegrino Collection</h3>
                    <p><strong>355 artifacts</strong></p>
                    <p>Ceramics from Masafi-5, Dibba, and Tell Abraq sites, spanning Late Bronze Age and Iron Age I.</p>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Chronological Framework</h2>
            <div class="periods-list">
                <span class="period-badge">Umm an-Nar (2700-2000 BCE)</span>
                <span class="period-badge">Wadi Suq (2000-1600 BCE)</span>
                <span class="period-badge">Late Bronze Age (1600-1250 BCE)</span>
                <span class="period-badge">Iron Age I-II (1250-300 BCE)</span>
            </div>
        </div>

        <div class="section">
            <h2>ML & API Access</h2>
            <p>Public REST API endpoints are available for machine learning applications and research integration:</p>
            <div class="api-section">
                <div class="api-endpoint">
                    <span class="method">GET</span> <span class="path">/api/v1/items</span>
                    <div class="desc">Retrieve all artifacts with full metadata (JSON)</div>
                </div>
                <div class="api-endpoint">
                    <span class="method">GET</span> <span class="path">/api/v1/items/{id}</span>
                    <div class="desc">Get single artifact by ID</div>
                </div>
                <div class="api-endpoint">
                    <span class="method">GET</span> <span class="path">/api/v1/vocabulary</span>
                    <div class="desc">Get controlled vocabulary terms for classification</div>
                </div>
                <div class="api-endpoint">
                    <span class="method">GET</span> <span class="path">/api/v1/periods</span>
                    <div class="desc">List all chronological periods in database</div>
                </div>
                <div class="api-endpoint">
                    <span class="method">GET</span> <span class="path">/api/v1/stats</span>
                    <div class="desc">Database statistics and counts</div>
                </div>
                <div class="api-endpoint">
                    <span class="method">GET</span> <span class="path">/{collection}/{folder}/{image}.png</span>
                    <div class="desc">Direct image access via image_path field</div>
                </div>
            </div>
        </div>

        <div class="login-section">
            <h3>Access Database</h3>
            <form class="login-form" onsubmit="login(event)">
                <input type="password" id="password" placeholder="Enter password" autocomplete="current-password">
                <button type="submit">Login</button>
                <p class="error" id="error">Invalid password</p>
            </form>
            <p class="role-info">
                <strong>Admin:</strong> Full access (edit, delete, rotate)<br>
                <strong>Viewer:</strong> Browse and search only
            </p>
        </div>

        <div class="footer">
            <p>Developed by <a href="https://github.com/enzococca" target="_blank">Enzo Cocca</a></p>
            <p style="margin-top:10px;font-size:0.8em;">
                <a href="https://github.com/enzococca/pottery-comparison" target="_blank">GitHub Repository</a>
            </p>
        </div>
    </div>

    <script>
        // Load live stats
        fetch('/api/v1/stats')
            .then(r => r.json())
            .then(stats => {
                document.getElementById('stats').innerHTML = `
                    <div class="stat-card"><div class="stat-number">${stats.total_items}</div><div class="stat-label">Artifacts</div></div>
                    <div class="stat-card"><div class="stat-number">${Object.keys(stats.by_collection).length}</div><div class="stat-label">Collections</div></div>
                    <div class="stat-card"><div class="stat-number">${Object.keys(stats.by_macro_period).length}</div><div class="stat-label">Periods</div></div>
                `;
            }).catch(() => {});

        async function login(e) {
            e.preventDefault();
            const password = document.getElementById('password').value;
            const error = document.getElementById('error');

            try {
                const response = await fetch('/api/login', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({password})
                });
                const result = await response.json();

                if (result.success) {
                    window.location.href = '/viewer';
                } else {
                    error.classList.add('show');
                    document.getElementById('password').value = '';
                }
            } catch (err) {
                error.textContent = 'Connection error';
                error.classList.add('show');
            }
        }
    </script>
</body>
</html>
'''


def get_viewer_html(role):
    """Generate viewer HTML with role-based UI"""
    is_admin = role == 'admin'
    admin_buttons = '''
        <button class="action-btn edit-btn" onclick="openEditModal()">&#9998; Edit</button>
        <button class="action-btn batch-btn" id="batchEditBtn" onclick="openBatchEditModal()">&#9998; Edit Sel.</button>
        <button class="action-btn batch-btn" id="batchDeleteBtn" onclick="confirmBatchDelete()">&#128465; Delete Sel.</button>
        <button class="action-btn delete-btn" onclick="confirmDelete()">&#128465;</button>
    ''' if is_admin else ''

    rotate_buttons = '''
        <div class="rotate-btns">
            <button class="rotate-btn" onclick="rotateImage(-90)" title="Rotate left">&#8634;</button>
            <button class="rotate-btn" onclick="rotateImage(90)" title="Rotate right">&#8635;</button>
            <button class="rotate-btn" onclick="flipImage('horizontal')" title="Flip horizontal (mirror)">&#8644;</button>
            <button class="rotate-btn" onclick="flipImage('vertical')" title="Flip vertical">&#8645;</button>
        </div>
    ''' if is_admin else ''

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CeramicaDatabase - Viewer</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }}
        .header {{
            background: rgba(0, 0, 0, 0.4);
            padding: 12px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .header h1 {{
            color: #4fc3f7;
            font-size: 1.4em;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .header-buttons {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
        .user-badge {{
            background: {'#4caf50' if is_admin else '#2196f3'};
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.75em;
            margin-right: 10px;
        }}
        .collection-tabs {{ display: flex; gap: 5px; flex-wrap: wrap; }}
        .collection-tab {{
            padding: 8px 16px;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.85em;
            transition: all 0.2s ease;
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
        }}
        .collection-tab.active {{
            background: var(--tab-color, #4fc3f7);
            color: #000;
            font-weight: bold;
        }}
        .collection-tab:hover:not(.active) {{ background: rgba(255, 255, 255, 0.2); }}
        .controls {{
            background: rgba(0, 0, 0, 0.2);
            padding: 10px 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .control-group {{ display: flex; align-items: center; gap: 6px; }}
        .controls label {{ color: #888; font-size: 0.75em; }}
        .controls select, .controls input {{
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #fff;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 0.8em;
        }}
        .stats {{
            margin-left: auto;
            background: rgba(79, 195, 247, 0.15);
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.85em;
            color: #4fc3f7;
        }}
        .main-container {{ display: flex; height: calc(100vh - 130px); }}
        .sidebar {{
            width: 280px;
            background: rgba(0, 0, 0, 0.25);
            overflow-y: auto;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .item-list {{ padding: 8px; }}
        .item-card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 8px;
            margin-bottom: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
            border-left: 3px solid transparent;
            display: flex;
            gap: 8px;
        }}
        .item-card:hover {{ background: rgba(255, 255, 255, 0.1); }}
        .item-card.active {{
            background: rgba(79, 195, 247, 0.15);
            border-left-color: var(--collection-color, #4fc3f7);
        }}
        .item-card img {{
            width: 50px;
            height: 50px;
            object-fit: cover;
            border-radius: 4px;
            flex-shrink: 0;
        }}
        .thumb-placeholder {{
            width: 50px;
            height: 50px;
            border-radius: 4px;
            flex-shrink: 0;
            background: linear-gradient(90deg, #2a2a2a 25%, #3a3a3a 50%, #2a2a2a 75%);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
        }}
        @keyframes shimmer {{
            0% {{ background-position: -200% 0; }}
            100% {{ background-position: 200% 0; }}
        }}
        .item-card .info {{ flex: 1; min-width: 0; overflow: hidden; }}
        .item-card .id {{
            font-weight: 600;
            color: #fff;
            font-size: 0.8em;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .item-card .period {{
            font-size: 0.7em;
            color: #ff9800;
            margin-top: 2px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .item-card .tags {{ display: flex; gap: 3px; margin-top: 3px; flex-wrap: wrap; }}
        .tag {{
            font-size: 0.55em;
            padding: 1px 5px;
            border-radius: 8px;
            background: rgba(255,255,255,0.1);
        }}
        .tag.decorated {{ background: #e91e63; color: white; }}
        .tag.plain {{ background: #607d8b; color: white; }}
        .tag.vessel {{ background: #2196f3; color: white; }}
        .content {{ flex: 1; display: flex; flex-direction: column; overflow-y: auto; }}
        .image-viewer {{
            flex-shrink: 0;
            min-height: 300px;
            max-height: calc(100vh - 350px);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 15px;
            background: rgba(0, 0, 0, 0.3);
            position: relative;
        }}
        .image-viewer img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
            transition: opacity 0.3s ease;
        }}
        .image-viewer img.loading {{
            opacity: 0.3;
        }}
        .image-spinner {{
            position: absolute;
            width: 40px;
            height: 40px;
            border: 4px solid rgba(79, 195, 247, 0.2);
            border-top-color: #4fc3f7;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            z-index: 10;
        }}
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        .nav-btn {{
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            background: rgba(79, 195, 247, 0.2);
            border: none;
            color: #4fc3f7;
            font-size: 1.8em;
            padding: 12px 10px;
            cursor: pointer;
            transition: all 0.2s ease;
            border-radius: 5px;
        }}
        .nav-btn:hover {{ background: rgba(79, 195, 247, 0.4); }}
        .nav-btn.prev {{ left: 10px; }}
        .nav-btn.next {{ right: 10px; }}
        .rotate-btns {{
            position: absolute;
            top: 10px;
            right: 10px;
            display: flex;
            gap: 5px;
        }}
        .rotate-btn {{
            background: rgba(156, 39, 176, 0.6);
            border: none;
            color: white;
            padding: 8px 12px;
            cursor: pointer;
            border-radius: 5px;
            font-size: 1em;
        }}
        .rotate-btn:hover {{ background: rgba(156, 39, 176, 0.9); }}

        /* Measurement Tool Styles */
        .measure-toolbar {{
            position: absolute;
            top: 10px;
            left: 10px;
            display: flex;
            gap: 5px;
            z-index: 100;
        }}
        .measure-btn {{
            background: rgba(76, 175, 80, 0.7);
            border: none;
            color: white;
            padding: 8px 12px;
            cursor: pointer;
            border-radius: 5px;
            font-size: 0.85em;
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .measure-btn:hover {{ background: rgba(76, 175, 80, 0.95); }}
        .measure-btn.active {{ background: #4caf50; box-shadow: 0 0 10px rgba(76, 175, 80, 0.5); }}
        .measure-btn.calibrate {{ background: rgba(255, 152, 0, 0.7); }}
        .measure-btn.calibrate:hover {{ background: rgba(255, 152, 0, 0.95); }}
        .measure-btn.calibrate.active {{ background: #ff9800; }}

        .measure-canvas {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 50;
        }}
        .measure-canvas.active {{
            pointer-events: auto;
            cursor: crosshair;
        }}

        .measure-info {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 5px;
            font-size: 0.85em;
            z-index: 100;
            max-width: 300px;
        }}
        .measure-info strong {{ color: #4fc3f7; }}

        .calibration-modal {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }}
        .calibration-modal.active {{ display: flex; }}
        .calibration-content {{
            background: #1a1a2e;
            padding: 25px;
            border-radius: 10px;
            max-width: 400px;
            text-align: center;
        }}
        .calibration-content h3 {{ margin-top: 0; color: #4fc3f7; }}
        .calibration-content input {{
            width: 100px;
            padding: 8px;
            margin: 10px 5px;
            border-radius: 5px;
            border: 1px solid #444;
            background: #0d0d1a;
            color: white;
            text-align: center;
        }}
        .calibration-content select {{
            padding: 8px;
            border-radius: 5px;
            border: 1px solid #444;
            background: #0d0d1a;
            color: white;
        }}
        .calibration-content .btn-row {{
            margin-top: 15px;
            display: flex;
            gap: 10px;
            justify-content: center;
        }}

        .metadata-panel {{
            background: rgba(0, 0, 0, 0.4);
            padding: 12px 15px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            max-height: 220px;
            overflow-y: auto;
        }}
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 8px;
        }}
        .meta-item {{
            background: rgba(255, 255, 255, 0.05);
            padding: 8px 10px;
            border-radius: 5px;
            border-left: 3px solid #4fc3f7;
        }}
        .meta-item .label {{
            font-size: 0.65em;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 2px;
        }}
        .meta-item .value {{
            font-size: 0.85em;
            color: #fff;
            word-break: break-word;
        }}
        .meta-item.period {{ border-left-color: #ff9800; }}
        .meta-item.period .value {{ color: #ff9800; font-weight: bold; }}
        .meta-item.decoration {{ border-left-color: #e91e63; }}
        .meta-item.decoration .value {{ color: #e91e63; }}
        .meta-item.vessel_type {{ border-left-color: #2196f3; }}
        .meta-item.vessel_type .value {{ color: #2196f3; }}
        .meta-item.page-ref {{ border-left-color: #4caf50; }}
        .meta-item.page-ref .value {{ color: #4caf50; cursor: pointer; text-decoration: underline; }}
        .action-btn {{
            border: none;
            color: white;
            padding: 6px 12px;
            border-radius: 18px;
            cursor: pointer;
            font-size: 0.75em;
            transition: all 0.2s ease;
        }}
        .action-btn:hover {{ transform: scale(1.05); }}
        .pdf-btn {{ background: linear-gradient(135deg, #4caf50, #45a049); }}
        .ml-btn {{ background: linear-gradient(135deg, #9c27b0, #6a1b9a); }}
        .delete-btn {{ background: linear-gradient(135deg, #f44336, #d32f2f); }}
        .select-btn {{ background: linear-gradient(135deg, #9c27b0, #7b1fa2); }}
        .select-btn.active {{ background: linear-gradient(135deg, #ff9800, #f57c00); }}
        .edit-btn {{ background: linear-gradient(135deg, #2196f3, #1976d2); }}
        .batch-btn {{ background: linear-gradient(135deg, #ff5722, #e64a19); display: none; }}
        .batch-btn.visible {{ display: inline-block; }}
        .logout-btn {{ background: linear-gradient(135deg, #607d8b, #455a64); }}

        /* ML Classifier Modal */
        .ml-modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            z-index: 2000;
            flex-direction: column;
        }}
        .ml-modal.active {{ display: flex; }}
        .ml-header {{
            background: linear-gradient(135deg, #9c27b0, #6a1b9a);
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .ml-header h2 {{ color: white; margin: 0; font-size: 1.3em; }}
        .ml-close {{
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            padding: 8px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
        }}
        .ml-close:hover {{ background: rgba(255,255,255,0.3); }}
        .ml-content {{
            flex: 1;
            display: flex;
            overflow: hidden;
            padding: 20px;
            gap: 20px;
        }}
        .ml-upload-section {{
            width: 400px;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 15px;
            max-height: calc(100vh - 150px);
            overflow-y: auto;
        }}
        .ml-drop-zone {{
            border: 3px dashed rgba(156, 39, 176, 0.5);
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: rgba(156, 39, 176, 0.1);
        }}
        .ml-drop-zone:hover, .ml-drop-zone.dragover {{
            border-color: #9c27b0;
            background: rgba(156, 39, 176, 0.2);
        }}
        .ml-drop-zone p {{ color: #aaa; margin: 10px 0; }}
        .ml-drop-zone .icon {{ font-size: 3em; color: #9c27b0; }}
        .ml-preview {{
            max-width: 100%;
            max-height: 200px;
            border-radius: 8px;
            display: none;
        }}
        .ml-preview.visible {{ display: block; margin: 10px auto; }}
        /* Drawing tools */
        .ml-image-container {{
            position: relative;
            display: inline-block;
            margin: 10px auto;
        }}
        .ml-image-container img {{
            max-width: 100%;
            max-height: 250px;
            border-radius: 8px;
            display: block;
        }}
        .ml-draw-canvas {{
            position: absolute;
            top: 0;
            left: 0;
            border-radius: 8px;
            cursor: crosshair;
        }}
        .ml-draw-tools {{
            display: none;
            gap: 8px;
            margin: 10px 0;
            flex-wrap: wrap;
            align-items: center;
        }}
        .ml-draw-tools.visible {{ display: flex; }}
        .draw-tool-btn {{
            padding: 6px 12px;
            border: 2px solid #444;
            background: rgba(255,255,255,0.1);
            color: #fff;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85em;
            transition: all 0.2s;
        }}
        .draw-tool-btn:hover {{ background: rgba(255,255,255,0.2); }}
        .draw-tool-btn.active {{
            border-color: #9c27b0;
            background: rgba(156, 39, 176, 0.3);
        }}
        .draw-tool-btn.clear {{
            border-color: #f44336;
            color: #f44336;
        }}
        .draw-color-picker {{
            display: flex;
            gap: 5px;
            align-items: center;
        }}
        .draw-color {{
            width: 24px;
            height: 24px;
            border-radius: 50%;
            cursor: pointer;
            border: 2px solid transparent;
            transition: transform 0.2s;
        }}
        .draw-color:hover {{ transform: scale(1.1); }}
        .draw-color.active {{ border-color: white; }}
        .draw-size-slider {{
            width: 80px;
            accent-color: #9c27b0;
        }}
        .ml-roi-indicator {{
            background: rgba(156, 39, 176, 0.2);
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 0.8em;
            color: #ce93d8;
            display: none;
        }}
        .ml-roi-indicator.visible {{ display: inline-block; }}
        .ml-classify-btn {{
            background: linear-gradient(135deg, #9c27b0, #6a1b9a);
            border: none;
            color: white;
            padding: 12px 25px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s;
        }}
        .ml-classify-btn:hover {{ transform: scale(1.02); }}
        .ml-classify-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
        .ml-explain-btn {{
            background: linear-gradient(135deg, #ff9800, #f57c00);
            border: none;
            color: white;
            padding: 12px 25px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s;
            margin-left: 10px;
        }}
        .ml-explain-btn:hover {{ transform: scale(1.02); }}
        .ml-explain-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
        .ml-explanation-panel {{
            background: rgba(0, 0, 0, 0.4);
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
            display: none;
            max-height: 500px;
            overflow-y: auto;
        }}
        .ml-explanation-panel.visible {{ display: block; }}
        .ml-explanation-panel h4 {{
            color: #ff9800;
            margin: 0 0 15px 0;
            font-size: 1.1em;
        }}
        .ml-heatmap-container {{
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
        }}
        .ml-heatmap-container img {{
            max-width: 200px;
            border-radius: 8px;
            border: 2px solid #333;
        }}
        .ml-predictions {{
            flex: 1;
        }}
        .ml-pred-item {{
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
            border-left: 4px solid #4fc3f7;
        }}
        .ml-pred-item.period {{ border-left-color: #ff9800; }}
        .ml-pred-item.decoration {{ border-left-color: #e91e63; }}
        .ml-pred-item.vessel {{ border-left-color: #2196f3; }}
        .ml-pred-item .pred-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
        }}
        .ml-pred-item .pred-label {{
            font-size: 0.75em;
            color: #888;
            text-transform: uppercase;
        }}
        .ml-pred-item .pred-value {{
            font-weight: bold;
            font-size: 1.1em;
        }}
        .ml-pred-item .pred-conf {{
            font-size: 0.9em;
            color: #4caf50;
        }}
        .ml-pred-item .pred-bar {{
            height: 4px;
            background: rgba(255,255,255,0.1);
            border-radius: 2px;
            margin-top: 5px;
        }}
        .ml-pred-item .pred-bar-fill {{
            height: 100%;
            border-radius: 2px;
            transition: width 0.5s ease;
        }}
        .ml-pred-item.period .pred-bar-fill {{ background: #ff9800; }}
        .ml-pred-item.decoration .pred-bar-fill {{ background: #e91e63; }}
        .ml-pred-item.vessel .pred-bar-fill {{ background: #2196f3; }}
        .ml-explanation-text {{
            background: rgba(255, 152, 0, 0.1);
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
            border-left: 4px solid #ff9800;
        }}
        .ml-explanation-text h5 {{
            color: #ff9800;
            margin: 0 0 10px 0;
            font-size: 0.95em;
        }}
        .ml-explanation-text p {{
            color: #ccc;
            font-size: 0.9em;
            line-height: 1.6;
            margin: 8px 0;
        }}
        .ml-explanation-text .focus-area {{
            color: #4fc3f7;
            font-style: italic;
        }}
        /* Explain Modal */
        .explain-modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.95);
            z-index: 3000;
            overflow-y: auto;
        }}
        .explain-modal.active {{ display: flex; flex-direction: column; }}
        .explain-modal-header {{
            background: linear-gradient(135deg, #ff9800, #f57c00);
            padding: 15px 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        .explain-modal-header h2 {{ color: white; margin: 0; font-size: 1.3em; }}
        .explain-modal-close {{
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            padding: 8px 15px;
            border-radius: 5px;
            cursor: pointer;
        }}
        .explain-modal-content {{
            flex: 1;
            padding: 25px;
            max-width: 900px;
            margin: 0 auto;
            width: 100%;
        }}
        .explain-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
            margin-bottom: 25px;
        }}
        .explain-image-section {{
            text-align: center;
        }}
        .explain-image-section img {{
            max-width: 100%;
            max-height: 300px;
            border-radius: 12px;
            border: 3px solid #333;
        }}
        .explain-image-section p {{
            color: #888;
            font-size: 0.85em;
            margin-top: 10px;
        }}
        .explain-predictions {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        .explain-pred-card {{
            background: rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 15px 20px;
            border-left: 5px solid #4fc3f7;
        }}
        .explain-pred-card.period {{ border-left-color: #ff9800; }}
        .explain-pred-card.decoration {{ border-left-color: #e91e63; }}
        .explain-pred-card.vessel {{ border-left-color: #2196f3; }}
        .explain-pred-card .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        .explain-pred-card .card-label {{
            color: #888;
            font-size: 0.8em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .explain-pred-card .card-conf {{
            font-size: 1.1em;
            font-weight: bold;
            color: #4caf50;
        }}
        .explain-pred-card .card-value {{
            font-size: 1.4em;
            font-weight: bold;
            color: white;
            margin-bottom: 8px;
        }}
        .explain-pred-card .card-bar {{
            height: 6px;
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
            overflow: hidden;
        }}
        .explain-pred-card .card-bar-fill {{
            height: 100%;
            border-radius: 3px;
            transition: width 0.8s ease;
        }}
        .explain-pred-card.period .card-bar-fill {{ background: linear-gradient(90deg, #ff9800, #ffb74d); }}
        .explain-pred-card.decoration .card-bar-fill {{ background: linear-gradient(90deg, #e91e63, #f06292); }}
        .explain-pred-card.vessel .card-bar-fill {{ background: linear-gradient(90deg, #2196f3, #64b5f6); }}
        .explain-reasons {{
            background: rgba(255, 152, 0, 0.1);
            border-radius: 12px;
            padding: 20px;
            border-left: 5px solid #ff9800;
        }}
        .explain-reasons h3 {{
            color: #ff9800;
            margin: 0 0 15px 0;
            font-size: 1.1em;
        }}
        .explain-reasons p {{
            color: #ddd;
            line-height: 1.7;
            margin: 12px 0;
            font-size: 0.95em;
        }}
        .explain-reasons .focus-highlight {{
            background: rgba(79, 195, 247, 0.15);
            padding: 12px 15px;
            border-radius: 8px;
            margin-top: 15px;
            border-left: 3px solid #4fc3f7;
        }}
        .explain-reasons .focus-highlight p {{
            color: #4fc3f7;
            margin: 0;
        }}
        /* Heatmap toggle on similar items */
        .ml-match-card {{
            position: relative;
        }}
        .ml-match-card .heatmap-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 0;
            transition: opacity 0.3s;
            pointer-events: none;
            border-radius: 8px;
        }}
        .ml-match-card.show-heatmap .heatmap-overlay {{
            opacity: 0.7;
        }}
        .heatmap-toggle-btn {{
            background: linear-gradient(135deg, #ff5722, #e64a19);
            border: none;
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.85em;
            margin-left: 10px;
        }}
        .heatmap-toggle-btn.active {{
            background: linear-gradient(135deg, #4caf50, #388e3c);
        }}
        .ml-results {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            display: none;
        }}
        .ml-results.visible {{ display: block; }}
        .ml-results h3 {{ color: #9c27b0; margin-bottom: 15px; }}
        .ml-result-item {{
            background: rgba(255,255,255,0.08);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
        }}
        .ml-result-item .label {{ color: #888; font-size: 0.8em; margin-bottom: 5px; }}
        .ml-result-item .value {{ color: #fff; font-size: 1.2em; font-weight: bold; }}
        .ml-result-item .confidence {{
            margin-top: 8px;
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            height: 10px;
            overflow: hidden;
        }}
        .ml-result-item .confidence-bar {{
            height: 100%;
            background: linear-gradient(90deg, #9c27b0, #e91e63);
            border-radius: 10px;
            transition: width 0.5s ease;
        }}
        .ml-result-item .confidence-text {{ font-size: 0.75em; color: #aaa; margin-top: 4px; }}
        .ml-threshold {{
            margin-top: 15px;
            padding: 15px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
        }}
        .ml-threshold label {{ color: #aaa; font-size: 0.85em; }}
        .ml-threshold input[type="range"] {{
            width: 100%;
            margin: 10px 0;
            accent-color: #9c27b0;
        }}
        .ml-threshold .threshold-value {{ color: #9c27b0; font-weight: bold; }}
        .ml-filter-btn {{
            background: linear-gradient(135deg, #4caf50, #45a049);
            border: none;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            margin-top: 10px;
            width: 100%;
        }}
        .ml-filter-btn:hover {{ transform: scale(1.02); }}
        .ml-matches-section {{
            flex: 1;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            overflow-y: auto;
        }}
        .ml-matches-section h3 {{ color: #4caf50; margin-bottom: 15px; }}
        .ml-matches-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 15px;
        }}
        .ml-match-card {{
            background: rgba(255,255,255,0.08);
            border-radius: 8px;
            padding: 10px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .ml-match-card:hover {{ background: rgba(255,255,255,0.15); transform: translateY(-3px); }}
        .ml-match-card img {{
            width: 100%;
            height: 120px;
            object-fit: contain;
            border-radius: 5px;
            background: rgba(0,0,0,0.3);
        }}
        .ml-match-card .match-id {{ color: #fff; font-size: 0.8em; margin-top: 8px; }}
        .ml-match-card .match-period {{ color: #ff9800; font-size: 0.7em; }}
        .ml-match-card .match-confidence {{ color: #9c27b0; font-size: 0.75em; font-weight: bold; }}

        /* ML Carousel Animation */
        .ml-carousel-container {{
            position: relative;
            height: 120px;
            overflow: hidden;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            margin-bottom: 15px;
        }}
        .ml-carousel {{
            display: flex;
            gap: 10px;
            position: absolute;
            animation: carouselScroll 30s linear infinite;
        }}
        .ml-carousel.paused {{ animation-play-state: paused; }}
        .ml-carousel-item {{
            width: 80px;
            height: 100px;
            flex-shrink: 0;
            border-radius: 5px;
            overflow: hidden;
            transition: all 0.3s;
            opacity: 0.5;
        }}
        .ml-carousel-item.analyzing {{
            opacity: 1;
            transform: scale(1.1);
            box-shadow: 0 0 20px rgba(156, 39, 176, 0.8);
        }}
        .ml-carousel-item.matched {{
            opacity: 1;
            box-shadow: 0 0 15px rgba(76, 175, 80, 0.8);
        }}
        .ml-carousel-item img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            background: rgba(255,255,255,0.05);
        }}
        @keyframes carouselScroll {{
            0% {{ transform: translateX(0); }}
            100% {{ transform: translateX(-50%); }}
        }}
        .ml-progress {{
            height: 4px;
            background: rgba(255,255,255,0.1);
            border-radius: 2px;
            overflow: hidden;
            margin-bottom: 10px;
        }}
        .ml-progress-bar {{
            height: 100%;
            background: linear-gradient(90deg, #9c27b0, #e91e63);
            width: 0%;
            transition: width 0.3s;
        }}
        .ml-progress-text {{
            font-size: 0.75em;
            color: #888;
            text-align: center;
        }}

        /* ML Analysis Section */
        .ml-analysis {{
            background: linear-gradient(135deg, rgba(156, 39, 176, 0.1), rgba(233, 30, 99, 0.1));
            border: 1px solid rgba(156, 39, 176, 0.3);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .ml-analysis h3 {{
            color: #e91e63;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .ml-analysis-text {{
            color: #ddd;
            line-height: 1.7;
            font-size: 0.95em;
        }}
        .ml-analysis-text strong {{ color: #fff; }}
        .ml-references {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }}
        .ml-references h4 {{ color: #4caf50; font-size: 0.9em; margin-bottom: 10px; }}
        .ml-ref-link {{
            display: inline-block;
            background: rgba(76, 175, 80, 0.2);
            color: #4caf50;
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            margin: 3px;
            cursor: pointer;
            text-decoration: none;
        }}
        .ml-ref-link:hover {{ background: rgba(76, 175, 80, 0.4); }}

        /* ML Statistics Chart */
        .ml-stats-section {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        .ml-stats-section h4 {{ color: #4fc3f7; margin-bottom: 15px; }}
        .ml-chart-container {{
            height: 200px;
            position: relative;
        }}
        .ml-stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-top: 15px;
        }}
        .ml-stat-box {{
            background: rgba(0,0,0,0.3);
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }}
        .ml-stat-box .stat-value {{ font-size: 1.5em; font-weight: bold; color: #4fc3f7; }}
        .ml-stat-box .stat-label {{ font-size: 0.75em; color: #888; }}

        .no-image {{ color: #666; text-align: center; padding: 20px; }}
        .item-checkbox {{
            width: 18px;
            height: 18px;
            cursor: pointer;
            accent-color: #9c27b0;
            flex-shrink: 0;
            display: none;
        }}
        .select-mode .item-checkbox {{ display: block; }}
        .item-card.selected {{
            background: rgba(156, 39, 176, 0.25);
            border-left-color: #9c27b0;
        }}
        .selection-count {{
            background: rgba(156, 39, 176, 0.3);
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.75em;
            color: #ce93d8;
            display: none;
        }}
        .selection-count.visible {{ display: inline-block; }}
        .loading {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 30px;
            color: #4fc3f7;
        }}
        .loading-spinner {{
            width: 35px;
            height: 35px;
            border: 3px solid rgba(79, 195, 247, 0.2);
            border-top-color: #4fc3f7;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 12px;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

        /* Modal styles */
        .modal-overlay {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            z-index: 10000;
            align-items: center;
            justify-content: center;
        }}
        .modal-overlay.active {{ display: flex; }}
        .modal {{
            background: #2a2a4a;
            padding: 25px;
            border-radius: 12px;
            max-width: 500px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        }}
        .modal h3 {{ color: #4fc3f7; margin-bottom: 15px; }}
        .modal p {{ margin-bottom: 15px; color: #ccc; }}
        .modal-buttons {{ display: flex; gap: 10px; justify-content: center; margin-top: 20px; }}
        .modal-btn {{
            padding: 10px 25px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.2s;
        }}
        .modal-btn.cancel {{ background: #555; color: #fff; }}
        .modal-btn.confirm {{ background: #4caf50; color: #fff; }}
        .modal-btn.danger {{ background: #f44336; color: #fff; }}
        .modal-btn:hover {{ opacity: 0.85; }}
        /* Decoration Catalog */
        .cat-filter-btn {{
            padding: 6px 12px;
            border: 1px solid #555;
            background: #333;
            color: #ccc;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85em;
        }}
        .cat-filter-btn.active {{ background: #4a6fa5; color: #fff; border-color: #4a6fa5; }}
        .cat-filter-btn:hover {{ background: #444; }}
        .cat-item {{
            background: #2a2a2a;
            border: 2px solid #444;
            border-radius: 8px;
            padding: 10px;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .cat-item:hover {{ border-color: #4a6fa5; background: #333; }}
        .cat-item.selected {{ border-color: #4caf50; background: #2a3a2a; }}
        .cat-item-thumb {{
            width: 100%;
            height: 60px;
            background: #fff;
            border-radius: 4px;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }}
        .cat-item-thumb img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
        .cat-item-code {{ font-weight: bold; color: #4a6fa5; font-size: 1.1em; }}
        .cat-item-name {{ font-size: 0.8em; color: #aaa; text-align: center; margin-top: 4px; }}
        .cat-item-category {{ font-size: 0.7em; color: #666; margin-top: 2px; }}
        .cat-item-info {{ cursor: pointer; flex: 1; }}
        .cat-pdf-btn {{
            margin-top: 5px;
            padding: 4px 8px;
            font-size: 0.75em;
            background: #3a5a8a;
            color: #fff;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        .cat-pdf-btn:hover {{ background: #4a6fa5; }}
        .cat-item-period {{
            font-size: 0.65em;
            padding: 2px 6px;
            border-radius: 3px;
            margin-top: 3px;
            display: inline-block;
        }}
        .cat-item-period.umm-an-nar {{ background: #2d5a3d; color: #8fbc8f; }}
        .cat-item-period.iron-age {{ background: #5a3d2d; color: #deb887; }}
        .cat-item-period.multi-period {{ background: #3d3d5a; color: #b8b8d1; }}
        .edit-form {{ display: flex; flex-direction: column; gap: 12px; }}
        .edit-form label {{ color: #888; font-size: 0.8em; }}
        .edit-row {{ display: flex; gap: 10px; align-items: center; }}
        .edit-row label {{ width: 100px; flex-shrink: 0; }}
        .edit-row input, .edit-row select {{
            flex: 1;
            padding: 8px;
            border-radius: 5px;
            border: 1px solid rgba(255,255,255,0.2);
            background: rgba(255,255,255,0.1);
            color: white;
        }}
        /* Autocomplete */
        .autocomplete-wrapper {{ position: relative; flex: 1; }}
        .autocomplete-list {{
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: #3a3a5a;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 5px;
            max-height: 150px;
            overflow-y: auto;
            z-index: 10;
            display: none;
        }}
        .autocomplete-list.show {{ display: block; }}
        .autocomplete-item {{
            padding: 8px 12px;
            cursor: pointer;
            font-size: 0.9em;
        }}
        .autocomplete-item:hover {{ background: rgba(79,195,247,0.2); }}
        .autocomplete-item .count {{ color: #888; font-size: 0.8em; margin-left: 8px; }}

        /* PDF Viewer Modal */
        .pdf-modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            z-index: 2000;
            flex-direction: column;
        }}
        .pdf-modal.active {{ display: flex; }}
        .pdf-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 20px;
            background: #1a1a2e;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .pdf-title {{
            color: #4fc3f7;
            font-size: 1em;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .pdf-title .page-info {{
            background: rgba(79, 195, 247, 0.2);
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.9em;
        }}
        .pdf-controls {{
            display: flex;
            gap: 8px;
        }}
        .pdf-ctrl-btn {{
            background: rgba(255,255,255,0.1);
            border: none;
            color: white;
            padding: 8px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.85em;
            transition: all 0.2s;
        }}
        .pdf-ctrl-btn:hover {{ background: rgba(255,255,255,0.2); }}
        .pdf-ctrl-btn.primary {{ background: #4caf50; }}
        .pdf-ctrl-btn.close {{ background: #f44336; }}
        .pdf-container {{
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: auto;
            padding: 20px;
        }}
        .pdf-canvas-wrapper {{
            position: relative;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            background: white;
        }}
        #pdfCanvas {{
            display: block;
            max-width: 100%;
            max-height: calc(100vh - 100px);
        }}
        .pdf-loading {{
            color: #4fc3f7;
            font-size: 1.2em;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 15px;
        }}
        .pdf-loading .spinner {{
            width: 50px;
            height: 50px;
            border: 4px solid rgba(79, 195, 247, 0.2);
            border-top-color: #4fc3f7;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        .highlight-overlay {{
            position: absolute;
            background: rgba(255, 255, 0, 0.4);
            pointer-events: none;
            border: 2px solid #ffc107;
            border-radius: 3px;
            animation: pulse-highlight 1.5s ease-in-out infinite;
        }}
        @keyframes pulse-highlight {{
            0%, 100% {{ opacity: 0.4; }}
            50% {{ opacity: 0.7; }}
        }}
        .pdf-nav {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .pdf-nav-btn {{
            background: rgba(79, 195, 247, 0.2);
            border: none;
            color: #4fc3f7;
            width: 35px;
            height: 35px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 1.2em;
            transition: all 0.2s;
        }}
        .pdf-nav-btn:hover {{ background: rgba(79, 195, 247, 0.4); }}
        .pdf-nav-btn:disabled {{ opacity: 0.3; cursor: not-allowed; }}

        /* 3D Viewer Modal */
        .viewer3d-modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.95);
            z-index: 3000;
            flex-direction: column;
        }}
        .viewer3d-modal.active {{ display: flex; }}
        .viewer3d-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            background: #1a1a2e;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .viewer3d-title {{
            color: #4fc3f7;
            font-size: 1.2em;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .viewer3d-info {{
            color: #888;
            font-size: 0.85em;
        }}
        .viewer3d-controls {{
            display: flex;
            gap: 10px;
        }}
        .viewer3d-btn {{
            background: rgba(79, 195, 247, 0.2);
            border: none;
            color: #4fc3f7;
            padding: 8px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.85em;
            transition: all 0.2s;
        }}
        .viewer3d-btn:hover {{ background: rgba(79, 195, 247, 0.4); }}
        .viewer3d-btn.close {{ background: rgba(255, 100, 100, 0.2); color: #ff6b6b; }}
        .viewer3d-btn.close:hover {{ background: rgba(255, 100, 100, 0.4); }}
        .viewer3d-container {{
            flex: 1;
            position: relative;
            overflow: hidden;
        }}
        #viewer3dCanvas {{
            width: 100%;
            height: 100%;
        }}
        .viewer3d-loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            color: #4fc3f7;
        }}
        .viewer3d-loading .spinner {{
            width: 60px;
            height: 60px;
            border: 4px solid rgba(79, 195, 247, 0.2);
            border-top-color: #4fc3f7;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        .viewer3d-help {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.7);
        }}
        .viewer3d-panel {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.85);
            padding: 15px;
            border-radius: 10px;
            color: #fff;
            font-size: 0.85em;
            min-width: 200px;
        }}
        .viewer3d-panel h4 {{
            margin: 0 0 10px 0;
            color: #4fc3f7;
            font-size: 1em;
        }}
        .viewer3d-panel label {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 8px 0;
        }}
        .viewer3d-panel input[type="range"] {{
            width: 100px;
        }}
        .viewer3d-panel input[type="checkbox"] {{
            width: 18px;
            height: 18px;
        }}
        .viewer3d-panel input[type="color"] {{
            width: 40px;
            height: 25px;
            border: none;
            cursor: pointer;
        }}
        .viewer3d-panel hr {{
            border: none;
            border-top: 1px solid #444;
            margin: 10px 0;
            padding: 10px 20px;
            border-radius: 25px;
            color: #888;
            font-size: 0.85em;
        }}
        .viewer3d-btn.ai-btn {{
            background: linear-gradient(135deg, #9b59b6, #8e44ad);
        }}
        .viewer3d-btn.ai-btn:hover {{
            background: linear-gradient(135deg, #8e44ad, #7d3c98);
        }}
        .viewer3d-ai-info {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(155, 89, 182, 0.9);
            padding: 12px 15px;
            border-radius: 8px;
            font-size: 0.85em;
            max-width: 300px;
            z-index: 100;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }}
        .viewer3d-ai-info h5 {{
            margin: 0 0 10px 0;
            font-size: 1em;
            color: #fff;
        }}
        .viewer3d-ai-info .ai-field {{
            margin: 5px 0;
            font-size: 0.9em;
        }}
        .viewer3d-ai-info .ai-label {{
            color: rgba(255,255,255,0.7);
            font-size: 0.8em;
        }}
        .viewer3d-ai-info .ai-value {{
            color: #fff;
            font-weight: 500;
        }}
        .viewer3d-ai-info .ai-patterns {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-top: 5px;
        }}
        .viewer3d-ai-info .ai-pattern-tag {{
            background: rgba(255,255,255,0.2);
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
        }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
</head>
<body>
    <div class="header">
        <h1>&#127994; CeramicaDatabase</h1>
        <div class="collection-tabs" id="collectionTabs"></div>
        <div class="header-buttons">
            <span class="user-badge">{'ADMIN' if is_admin else 'VIEWER'}</span>
            <span class="selection-count" id="selectionCount">0 sel.</span>
            <button class="action-btn ml-btn" onclick="openMlClassifier()">&#129504; ML Classify</button>
            <button class="action-btn pdf-btn" onclick="openPdfAtPage()">&#128196; PDF</button>
            {'<button class="action-btn select-btn" id="selectBtn" onclick="toggleSelectMode()">&#9745; Select</button>' if is_admin else ''}
            {admin_buttons}
            <button class="action-btn logout-btn" onclick="logout()">Logout</button>
        </div>
    </div>
    <div class="controls">
        <div class="control-group">
            <label>Macro-Period:</label>
            <select id="filterMacroPeriod">
                <option value="">All</option>
                <option value="Umm an-Nar">Umm an-Nar</option>
                <option value="Wadi Suq">Wadi Suq</option>
                <option value="Late Bronze Age">Late Bronze Age</option>
            </select>
        </div>
        <div class="control-group">
            <label>Period:</label>
            <select id="filterPeriod">
                <option value="">All</option>
            </select>
        </div>
        <div class="control-group">
            <label>Decoration:</label>
            <select id="filterDecoration">
                <option value="">All</option>
            </select>
        </div>
        <div class="control-group">
            <label>Vessel Type:</label>
            <select id="filterVesselType">
                <option value="">All</option>
            </select>
        </div>
        <div class="control-group">
            <label>Part:</label>
            <select id="filterPartType">
                <option value="">All</option>
            </select>
        </div>
        <div class="control-group">
            <label>Search:</label>
            <input type="text" id="searchInput" placeholder="ID, type...">
        </div>
        <div class="stats" id="stats">Loading...</div>
    </div>
    <div class="main-container">
        <div class="sidebar">
            <div class="item-list" id="itemList">
                <div class="loading">
                    <div class="loading-spinner"></div>
                    <p>Loading data...</p>
                </div>
            </div>
        </div>
        <div class="content">
            <div class="image-viewer" id="imageViewer">
                <div class="no-image"><p>Select an item from the list</p></div>
                <button class="nav-btn prev" onclick="navigate(-1)">&#8249;</button>
                <button class="nav-btn next" onclick="navigate(1)">&#8250;</button>
                {rotate_buttons}
                <!-- Measurement Tool -->
                <div class="measure-toolbar" id="measureToolbar" style="display:none;">
                    <button class="measure-btn calibrate" id="calibrateBtn" onclick="startCalibration()" title="Calibrate scale">
                        &#128207; Calibrate
                    </button>
                    <button class="measure-btn" id="measureBtn" onclick="toggleMeasure()" title="Measure distance">
                        &#128207; Measure
                    </button>
                    <button class="measure-btn" id="clearMeasureBtn" onclick="clearMeasurements()" title="Clear measurements">
                        &#10006; Clear
                    </button>
                    <button class="measure-btn" id="view3dBtn" onclick="open3DViewer()" title="View 3D reconstruction">
                        &#127912; 3D
                    </button>
                </div>
                <canvas class="measure-canvas" id="measureCanvas"></canvas>
                <div class="measure-info" id="measureInfo" style="display:none;"></div>
            </div>
            <div class="metadata-panel">
                <div class="metadata-grid" id="metadataGrid"></div>
            </div>
        </div>
    </div>

    <!-- Calibration Modal -->
    <div class="calibration-modal" id="calibrationModal">
        <div class="calibration-content">
            <h3>&#128207; Calibration</h3>
            <p id="calibrationStep">Click two points on a known distance (e.g., scale bar)</p>
            <div id="calibrationInput" style="display:none;">
                <p>Enter the real distance between the two points:</p>
                <input type="number" id="calibrationDistance" step="0.1" min="0.1" value="5">
                <select id="calibrationUnit">
                    <option value="cm">cm</option>
                    <option value="mm">mm</option>
                    <option value="m">m</option>
                </select>
                <div class="btn-row">
                    <button class="modal-btn cancel" onclick="cancelCalibration()">Cancel</button>
                    <button class="modal-btn primary" onclick="applyCalibration()">Apply</button>
                </div>
            </div>
        </div>
    </div>

    <!-- 3D Viewer Modal -->
    <div class="viewer3d-modal" id="viewer3dModal">
        <div class="viewer3d-header">
            <div class="viewer3d-title">
                <span>&#127912; 3D Vessel Reconstruction</span>
                <span class="viewer3d-info" id="viewer3dInfo"></span>
            </div>
            <div class="viewer3d-controls">
                <button class="viewer3d-btn ai-btn" id="aiDecorationBtn" onclick="openAIModal()" title="Analyze decoration with Claude AI">&#129302; AI Analysis</button>
                <button class="viewer3d-btn" id="toggleDecorationBtn" onclick="toggleDecoration()" style="display:none;">&#127912; Plain</button>
                <button class="viewer3d-btn" id="toggleAIBtn" onclick="toggleAIDecoration()" style="display:none;">&#129302; AI</button>
                <button class="viewer3d-btn" onclick="reset3DView()">&#8635; Reset View</button>
                <button class="viewer3d-btn" onclick="download3D()">&#8595; Download GLB</button>
                <button class="viewer3d-btn close" onclick="close3DViewer()">&#10005; Close</button>
            </div>
        </div>
        <div class="viewer3d-container" id="viewer3dContainer">
            <div class="viewer3d-loading" id="viewer3dLoading">
                <div class="spinner"></div>
                <p>Generating 3D model...</p>
            </div>
            <!-- AI Analysis Info Panel -->
            <div class="viewer3d-ai-info" id="viewer3dAIInfo" style="display:none;">
                <h5>&#129302; AI Decoration Analysis</h5>
                <div id="aiAnalysisContent"></div>
            </div>
            <!-- Control Panel -->
            <div class="viewer3d-panel" id="viewer3dPanel">
                <h4>&#9881; Display Options</h4>
                <label>Color: <input type="color" id="v3dColor" value="#ffffff" onchange="update3DColor(this.value)"></label>
                <label>Wireframe: <input type="checkbox" id="v3dWireframe" onchange="toggle3DWireframe(this.checked)"></label>
                <label>Edges: <input type="checkbox" id="v3dEdges" checked onchange="toggle3DEdges(this.checked)"></label>
                <label>Edge Color: <input type="color" id="v3dEdgeColor" value="#000000" onchange="update3DEdgeColor(this.value)"></label>
                <hr>
                <label>Opacity: <input type="range" id="v3dOpacity" min="0.1" max="1" step="0.1" value="1" onchange="update3DOpacity(this.value)"></label>
                <label>Light: <input type="range" id="v3dLight" min="0.2" max="2" step="0.1" value="1" onchange="update3DLight(this.value)"></label>
                <hr>
                <label>Grid: <input type="checkbox" id="v3dGrid" checked onchange="toggle3DGrid(this.checked)"></label>
                <label>Auto Rotate: <input type="checkbox" id="v3dAutoRotate" onchange="toggle3DAutoRotate(this.checked)"></label>
            </div>
            <div class="viewer3d-help">&#128270; Drag to rotate | Scroll to zoom | Right-click to pan</div>
        </div>
    </div>

    <!-- AI API Key Modal -->
    <div class="modal-overlay" id="aiKeyModal">
        <div class="modal" style="max-width: 500px;">
            <h3>&#129302; Claude AI Decoration Analysis</h3>
            <p style="color: #aaa; margin-bottom: 15px;">Enter your Anthropic API key to analyze decorations with Claude AI.</p>
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px;">API Key:</label>
                <input type="password" id="anthropicApiKey" placeholder="sk-ant-..." style="width: 100%; padding: 10px; background: #333; border: 1px solid #555; border-radius: 4px; color: #fff;">
            </div>
            <div style="margin-bottom: 15px;">
                <label style="display: flex; align-items: center; gap: 8px;">
                    <input type="checkbox" id="saveApiKey" checked>
                    <span>Remember API key (stored locally)</span>
                </label>
            </div>
            <div class="modal-buttons">
                <button class="modal-btn cancel" onclick="closeModal('aiKeyModal')">Cancel</button>
                <button class="modal-btn confirm" onclick="runAIAnalysis()">&#129302; Analyze with AI</button>
            </div>
        </div>
    </div>

    <!-- Delete modal -->
    <div class="modal-overlay" id="deleteModal">
        <div class="modal">
            <h3>&#9888; Confirm Delete</h3>
            <p id="deleteMessage">Are you sure you want to delete?</p>
            <div class="modal-buttons">
                <button class="modal-btn cancel" onclick="closeModal('deleteModal')">Cancel</button>
                <button class="modal-btn danger" onclick="executeDelete()">Delete</button>
            </div>
        </div>
    </div>

    <!-- Edit modal -->
    <div class="modal-overlay" id="editModal">
        <div class="modal">
            <h3>&#9998; Edit Item</h3>
            <div class="edit-form">
                <div class="edit-row">
                    <label>Decoration:</label>
                    <div class="autocomplete-wrapper">
                        <input type="text" id="editDecoration" placeholder="Type or select...">
                        <div class="autocomplete-list" id="decorationList"></div>
                    </div>
                </div>
                <div class="edit-row">
                    <label>Vessel Type:</label>
                    <div class="autocomplete-wrapper">
                        <input type="text" id="editVesselType" placeholder="Type or select...">
                        <div class="autocomplete-list" id="vesselTypeList"></div>
                    </div>
                </div>
                <div class="edit-row">
                    <label>Part:</label>
                    <div class="autocomplete-wrapper">
                        <input type="text" id="editPartType" placeholder="Type or select...">
                        <div class="autocomplete-list" id="partTypeList"></div>
                    </div>
                </div>
                <div class="edit-row">
                    <label>Period:</label>
                    <div class="autocomplete-wrapper">
                        <input type="text" id="editPeriod" placeholder="Type or select...">
                        <div class="autocomplete-list" id="periodList"></div>
                    </div>
                </div>
                <div class="edit-row">
                    <label>Decoration Code:</label>
                    <div style="display:flex;gap:5px;flex:1;">
                        <input type="text" id="editDecorationCode" placeholder="e.g., 101, 124..." style="flex:1;padding:8px;border-radius:5px;border:1px solid #444;background:#333;color:#fff;">
                        <button type="button" onclick="openDecorationCatalog()" style="padding:8px 12px;background:#4a6fa5;color:#fff;border:none;border-radius:5px;cursor:pointer;" title="Browse decoration catalog">📖 Catalog</button>
                    </div>
                </div>
                <div class="edit-row">
                    <label>Decorative Motif:</label>
                    <div class="autocomplete-wrapper">
                        <input type="text" id="editMotivoDecorativo" placeholder="e.g., wavy lines, geometric...">
                        <div class="autocomplete-list" id="motivoDecorativoList"></div>
                    </div>
                </div>
                <div class="edit-row">
                    <label>Decorative Syntax:</label>
                    <div class="autocomplete-wrapper">
                        <input type="text" id="editSintassiDecorativa" placeholder="e.g., rim band, shoulder...">
                        <div class="autocomplete-list" id="sintassiDecorativaList"></div>
                    </div>
                </div>
                <div class="edit-row">
                    <label>Scale:</label>
                    <div class="autocomplete-wrapper">
                        <input type="text" id="editScalaMetrica" placeholder="e.g., 1:3">
                        <div class="autocomplete-list" id="scalaMetricaList"></div>
                    </div>
                </div>
                <div class="edit-row">
                    <label>Width (cm):</label>
                    <input type="number" id="editLarghezzaCm" placeholder="Actual width" step="0.1" style="flex:1;padding:8px;border-radius:5px;border:1px solid #444;background:#333;color:#fff;">
                </div>
                <div class="edit-row">
                    <label>Height (cm):</label>
                    <input type="number" id="editAltezzaCm" placeholder="Actual height" step="0.1" style="flex:1;padding:8px;border-radius:5px;border:1px solid #444;background:#333;color:#fff;">
                </div>
            </div>
            <div class="modal-buttons">
                <button class="modal-btn cancel" onclick="closeModal('editModal')">Cancel</button>
                <button class="modal-btn confirm" onclick="saveEdit()">Save</button>
            </div>
        </div>
    </div>

    <!-- Batch Edit modal -->
    <div class="modal-overlay" id="batchEditModal">
        <div class="modal">
            <h3>&#9998; Batch Edit (<span id="batchCount">0</span> items)</h3>
            <p style="font-size:0.8em;color:#888;">Leave empty to not change</p>
            <div class="edit-form">
                <div class="edit-row">
                    <label>Decoration:</label>
                    <div class="autocomplete-wrapper">
                        <input type="text" id="batchDecoration" placeholder="Type or select...">
                        <div class="autocomplete-list" id="batchDecorationList"></div>
                    </div>
                </div>
                <div class="edit-row">
                    <label>Vessel Type:</label>
                    <div class="autocomplete-wrapper">
                        <input type="text" id="batchVesselType" placeholder="Type or select...">
                        <div class="autocomplete-list" id="batchVesselTypeList"></div>
                    </div>
                </div>
                <div class="edit-row">
                    <label>Part:</label>
                    <div class="autocomplete-wrapper">
                        <input type="text" id="batchPartType" placeholder="Type or select...">
                        <div class="autocomplete-list" id="batchPartTypeList"></div>
                    </div>
                </div>
            </div>
            <div class="modal-buttons">
                <button class="modal-btn cancel" onclick="closeModal('batchEditModal')">Cancel</button>
                <button class="modal-btn confirm" onclick="saveBatchEdit()">Save All</button>
            </div>
        </div>
    </div>

    <!-- Decoration Catalog Modal -->
    <div class="modal-overlay" id="decorationCatalogModal" style="z-index:20000;">
        <div class="modal" style="max-width:900px;max-height:85vh;overflow:hidden;display:flex;flex-direction:column;">
            <h3>📖 Decoration Catalog</h3>
            <p style="font-size:0.8em;color:#888;margin:-10px 0 10px 0;">Source: Schmidt, Bat - Tab. 12 (Umm an-Nar period)</p>
            <div style="display:flex;gap:10px;margin-bottom:10px;flex-wrap:wrap;">
                <button class="cat-filter-btn active" onclick="filterCatalog('all')">All</button>
                <button class="cat-filter-btn" onclick="filterCatalog('bemalung')">Paintings</button>
                <button class="cat-filter-btn" onclick="filterCatalog('kombination')">Combinations</button>
                <button class="cat-filter-btn" onclick="filterCatalog('negativ')">Incised</button>
                <button class="cat-filter-btn" onclick="filterCatalog('positiv')">Relief</button>
            </div>
            <input type="text" id="catalogSearch" placeholder="Search by code or name..."
                   style="width:100%;padding:10px;margin-bottom:10px;border-radius:5px;border:1px solid #444;background:#333;color:#fff;"
                   oninput="searchCatalog(this.value)">
            <div id="catalogGrid" style="flex:1;overflow-y:auto;display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px;padding:10px 0;">
                <!-- Catalog items will be loaded here -->
            </div>
            <div class="modal-buttons">
                <button class="modal-btn cancel" onclick="closeModal('decorationCatalogModal')">Close</button>
            </div>
        </div>
    </div>

    <!-- PDF Viewer Modal -->
    <div class="pdf-modal" id="pdfModal">
        <div class="pdf-header">
            <div class="pdf-title">
                <span id="pdfFileName">PDF Viewer</span>
                <span class="page-info">Page <span id="pdfCurrentPage">1</span> / <span id="pdfTotalPages">?</span></span>
            </div>
            <div class="pdf-controls">
                <div class="pdf-nav">
                    <button class="pdf-nav-btn" onclick="pdfGoToPage(-1)" id="pdfPrevBtn">&#8249;</button>
                    <button class="pdf-nav-btn" onclick="pdfGoToPage(1)" id="pdfNextBtn">&#8250;</button>
                </div>
                <button class="pdf-ctrl-btn" onclick="pdfZoom(-0.25)">&#8722; Zoom</button>
                <button class="pdf-ctrl-btn" onclick="pdfZoom(0.25)">&#43; Zoom</button>
                <button class="pdf-ctrl-btn primary" onclick="downloadPdfPage()">&#8681; Save Page</button>
                <button class="pdf-ctrl-btn" onclick="openFullPdf()">&#8599; Open Full PDF</button>
                <button class="pdf-ctrl-btn close" onclick="closePdfModal()">&#10005; Close</button>
            </div>
        </div>
        <div class="pdf-container" id="pdfContainer">
            <div class="pdf-loading" id="pdfLoading">
                <div class="spinner"></div>
                <span>Loading PDF page...</span>
            </div>
            <div class="pdf-canvas-wrapper" id="pdfCanvasWrapper" style="display:none;">
                <canvas id="pdfCanvas"></canvas>
            </div>
        </div>
    </div>

    <!-- ML Classifier Modal -->
    <div class="ml-modal" id="mlModal">
        <div class="ml-header">
            <h2>&#129504; Visual Similarity Search</h2>
            <button class="ml-close" onclick="closeMlModal()">&#10005; Close</button>
        </div>
        <div class="ml-content">
            <div class="ml-upload-section">
                <h3 style="color: #9c27b0; margin-bottom: 10px;">Upload Ceramic Image</h3>
                <div class="ml-drop-zone" id="mlDropZone" onclick="document.getElementById('mlFileInput').click()">
                    <div class="icon">&#128247;</div>
                    <p>Click or drag an image here</p>
                    <p style="font-size: 0.8em;">Supports JPG, PNG</p>
                </div>
                <input type="file" id="mlFileInput" accept="image/*" style="display: none;" onchange="handleMlFile(event)">

                <!-- Image with drawing canvas overlay -->
                <div class="ml-image-container" id="mlImageContainer" style="display: none;">
                    <img class="ml-preview" id="mlPreview">
                    <canvas class="ml-draw-canvas" id="mlDrawCanvas"></canvas>
                </div>

                <!-- Drawing tools -->
                <div class="ml-draw-tools" id="mlDrawTools">
                    <button class="draw-tool-btn" id="drawRectBtn" onclick="setDrawTool('rect')" title="Rettangolo">
                        &#9634; Rect
                    </button>
                    <button class="draw-tool-btn" id="drawFreeBtn" onclick="setDrawTool('free')" title="Disegno libero">
                        &#9998; Free
                    </button>
                    <button class="draw-tool-btn clear" onclick="clearDrawing()" title="Cancella">
                        &#128465; Clear
                    </button>
                    <span style="color: #888; font-size: 0.8em;">|</span>
                    <div class="draw-color-picker">
                        <div class="draw-color active" style="background: #ff5722;" onclick="setDrawColor('#ff5722')" title="Arancione"></div>
                        <div class="draw-color" style="background: #4caf50;" onclick="setDrawColor('#4caf50')" title="Verde"></div>
                        <div class="draw-color" style="background: #2196f3;" onclick="setDrawColor('#2196f3')" title="Blu"></div>
                        <div class="draw-color" style="background: #ffeb3b;" onclick="setDrawColor('#ffeb3b')" title="Giallo"></div>
                    </div>
                    <input type="range" class="draw-size-slider" id="drawSizeSlider" min="2" max="20" value="5"
                           oninput="setDrawSize(this.value)" title="Spessore">
                    <span class="ml-roi-indicator" id="roiIndicator">&#127919; ROI attiva</span>
                </div>

                <div class="ml-threshold" style="margin-top: 15px;">
                    <label>Similarity Threshold: <span class="threshold-value" id="thresholdValue">30%</span></label>
                    <input type="range" id="mlThreshold" min="0" max="100" value="30"
                           oninput="updateThreshold(this.value)">
                </div>

                <!-- Real Photo Toggle -->
                <div class="ml-photo-toggle" style="margin-top: 15px; padding: 10px; background: #1a1a2e; border-radius: 8px; border: 1px solid #333;">
                    <label style="display: flex; align-items: center; gap: 10px; cursor: pointer;">
                        <input type="checkbox" id="realPhotoToggle" onchange="toggleRealPhotoMode(this.checked)">
                        <span style="font-size: 1.1em;">&#128247;</span>
                        <span><strong>Foto Reale</strong></span>
                        <span style="color: #888; font-size: 0.85em;">(preprocessing)</span>
                    </label>

                    <div id="realPhotoOptions" style="display: none; margin-top: 15px;">
                        <!-- Parameter Sliders -->
                        <div style="margin-bottom: 15px; padding: 10px; background: #252540; border-radius: 5px;">
                            <p style="color: #aaa; font-size: 0.85em; margin-bottom: 10px;"><strong>Parametri Estrazione:</strong></p>

                            <div style="margin-bottom: 8px;">
                                <label style="color: #888; font-size: 0.8em; display: flex; justify-content: space-between;">
                                    <span>Soglia Contrasto:</span>
                                    <span id="thresholdValue">18</span>
                                </label>
                                <input type="range" id="contrastThreshold" min="5" max="50" value="18"
                                    style="width: 100%;" onchange="updatePreprocessParams()">
                            </div>

                            <div style="margin-bottom: 8px;">
                                <label style="color: #888; font-size: 0.8em; display: flex; justify-content: space-between;">
                                    <span>Area Minima (%):</span>
                                    <span id="minAreaValue">0.05</span>
                                </label>
                                <input type="range" id="minAreaSlider" min="0.01" max="0.5" step="0.01" value="0.05"
                                    style="width: 100%;" onchange="updatePreprocessParams()">
                            </div>

                            <div style="margin-bottom: 8px;">
                                <label style="color: #888; font-size: 0.8em; display: flex; justify-content: space-between;">
                                    <span>Blur:</span>
                                    <span id="blurValue">9</span>
                                </label>
                                <input type="range" id="blurSlider" min="1" max="25" step="2" value="9"
                                    style="width: 100%;" onchange="updatePreprocessParams()">
                            </div>

                            <button onclick="applyPreprocessParams()" style="width: 100%; padding: 5px; margin-top: 5px; background: #4a4a6a; border: none; color: white; border-radius: 4px; cursor: pointer;">
                                &#128260; Applica Parametri
                            </button>
                        </div>

                        <!-- Manual Drawing Section -->
                        <div style="padding: 10px; background: #252540; border-radius: 5px;">
                            <p style="color: #aaa; font-size: 0.85em; margin-bottom: 10px;"><strong>&#9999; Disegna Decorazioni:</strong></p>
                            <p style="color: #666; font-size: 0.75em; margin-bottom: 10px;">Disegna sopra le decorazioni per estrarle manualmente</p>

                            <div style="display: flex; gap: 5px; margin-bottom: 10px; flex-wrap: wrap;">
                                <button id="drawModeBtn" onclick="setDrawMode('draw')" style="padding: 5px 10px; background: #6a4a9a; border: none; color: white; border-radius: 4px; cursor: pointer;">
                                    &#9999; Disegna
                                </button>
                                <button id="eraseModeBtn" onclick="setDrawMode('erase')" style="padding: 5px 10px; background: #4a4a6a; border: none; color: white; border-radius: 4px; cursor: pointer;">
                                    &#128065; Cancella
                                </button>
                                <button onclick="clearDrawing()" style="padding: 5px 10px; background: #6a4a4a; border: none; color: white; border-radius: 4px; cursor: pointer;">
                                    &#128465; Reset
                                </button>
                            </div>

                            <div style="margin-bottom: 10px;">
                                <label style="color: #888; font-size: 0.8em;">Spessore pennello: <span id="brushSizeValue">15</span>px</label>
                                <input type="range" id="brushSize" min="5" max="50" value="15" style="width: 100%;">
                            </div>

                            <!-- Drawing Canvas Container -->
                            <div id="drawingContainer" style="position: relative; display: inline-block; border: 2px solid #444; border-radius: 5px; overflow: hidden;">
                                <img id="drawingBaseImage" style="max-width: 280px; display: block;">
                                <canvas id="drawingCanvas" style="position: absolute; top: 0; left: 0; cursor: crosshair;"></canvas>
                            </div>

                            <button onclick="openEnlargedDrawing()" style="width: 100%; padding: 6px; margin-top: 8px; background: #4a4a6a; border: none; color: white; border-radius: 4px; cursor: pointer;">
                                &#128269; Ingrandisci per Dettagli
                            </button>

                            <button onclick="applyManualDrawing()" style="width: 100%; padding: 8px; margin-top: 8px; background: #4a6a4a; border: none; color: white; border-radius: 4px; cursor: pointer; font-weight: bold;">
                                &#10004; Usa Disegno (+ contorno auto)
                            </button>
                        </div>

                        <!-- Preview -->
                        <div style="margin-top: 15px;">
                            <p style="color: #888; font-size: 0.8em; margin-bottom: 5px;">Preview conversione:</p>
                            <img id="preprocessedPreview" style="max-width: 200px; border-radius: 5px; border: 1px solid #444;">
                        </div>
                    </div>
                </div>

                <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 15px;">
                    <button class="ml-classify-btn" id="mlClassifyBtn" onclick="runSimilaritySearch()" disabled>
                        &#128269; Find Similar
                    </button>
                    <button class="ml-explain-btn" id="mlExplainBtn" onclick="runExplainClassification()" disabled>
                        &#129504; Classify & Explain
                    </button>
                </div>

                <!-- Explanation Panel -->
                <div class="ml-explanation-panel" id="mlExplanationPanel">
                    <h4>&#129504; AI Classification & Explanation</h4>
                    <div class="ml-heatmap-container">
                        <div>
                            <p style="color: #888; font-size: 0.8em; margin-bottom: 5px;">Grad-CAM Heatmap</p>
                            <img id="mlHeatmap" src="" alt="Heatmap">
                            <p style="color: #666; font-size: 0.7em; margin-top: 5px;">Zone rosse = aree di focus del modello</p>
                        </div>
                        <div class="ml-predictions" id="mlPredictions"></div>
                    </div>
                    <div class="ml-explanation-text" id="mlExplanationText"></div>
                </div>

                <!-- Carousel Section -->
                <div id="mlCarouselSection" style="display: none; margin-top: 20px;">
                    <div class="ml-progress">
                        <div class="ml-progress-bar" id="mlProgressBar"></div>
                    </div>
                    <div class="ml-progress-text" id="mlProgressText">Analyzing database...</div>
                    <div class="ml-carousel-container">
                        <div class="ml-carousel" id="mlCarousel"></div>
                    </div>
                </div>
            </div>

            <div class="ml-matches-section" id="mlMatchesSection">
                <!-- Analysis Section -->
                <div class="ml-analysis" id="mlAnalysis" style="display: none;">
                    <h3>&#128220; Archaeological Analysis</h3>
                    <div class="ml-analysis-text" id="mlAnalysisText"></div>
                    <div class="ml-references" id="mlReferences"></div>
                </div>

                <!-- Statistics Section -->
                <div class="ml-stats-section" id="mlStatsSection" style="display: none;">
                    <h4>&#128202; Distribution Statistics</h4>
                    <div class="ml-chart-container">
                        <canvas id="mlChart"></canvas>
                    </div>
                    <div class="ml-stats-grid" id="mlStatsGrid"></div>
                </div>

                <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
                    <h3 style="margin: 0;">&#128270; Visually Similar Ceramics <span id="mlMatchCount">(0)</span></h3>
                    <button class="heatmap-toggle-btn" id="heatmapToggleBtn" onclick="toggleHeatmaps()" style="display: none;">
                        &#128293; Show Heatmaps
                    </button>
                </div>
                <p style="color: #888; font-size: 0.85em; margin-bottom: 15px;">
                    Ranked by visual similarity based on decoration patterns
                </p>
                <div class="ml-matches-grid" id="mlMatchesGrid">
                    <p style="color: #666; text-align: center; grid-column: 1/-1;">
                        Upload an image to find visually similar ceramics
                    </p>
                </div>
            </div>
        </div>
    </div>

    <!-- Explain Modal (popup for AI explanation) -->
    <div class="explain-modal" id="explainModal">
        <div class="explain-modal-header">
            <h2>&#129504; AI Classification & Explanation</h2>
            <button class="explain-modal-close" onclick="closeExplainModal()">&#10005; Close</button>
        </div>
        <div class="explain-modal-content">
            <div class="explain-grid">
                <div class="explain-image-section">
                    <p style="margin-bottom: 8px; color: #aaa;">Original Image</p>
                    <img id="explainOriginal" src="" alt="Original">
                </div>
                <div class="explain-image-section">
                    <p style="margin-bottom: 8px; color: #ff5722;">Grad-CAM Heatmap</p>
                    <img id="explainHeatmap" src="" alt="Heatmap">
                    <p>&#128308; Zone rosse = aree di maggior attenzione del modello</p>
                </div>
            </div>
            <div class="explain-predictions" id="explainPredictions"></div>
            <div class="explain-reasons" id="explainReasons"></div>
        </div>
    </div>

    <!-- Enlarged Drawing Modal -->
    <div class="explain-modal" id="enlargedDrawingModal" style="z-index: 10001;">
        <div class="explain-modal-header">
            <h2>&#9999; Disegna Decorazioni (Vista Ingrandita)</h2>
            <button class="explain-modal-close" onclick="closeEnlargedDrawing()">&#10005; Chiudi</button>
        </div>
        <div class="explain-modal-content" style="padding: 20px;">
            <p style="color: #888; margin-bottom: 15px;">Disegna le decorazioni sull'immagine. Il contorno della ceramica verrà estratto automaticamente.</p>

            <div style="display: flex; gap: 10px; margin-bottom: 15px; flex-wrap: wrap; align-items: center;">
                <button id="enlargedDrawModeBtn" onclick="setEnlargedDrawMode('draw')" style="padding: 8px 15px; background: #6a4a9a; border: none; color: white; border-radius: 4px; cursor: pointer;">
                    &#9999; Disegna
                </button>
                <button id="enlargedEraseModeBtn" onclick="setEnlargedDrawMode('erase')" style="padding: 8px 15px; background: #4a4a6a; border: none; color: white; border-radius: 4px; cursor: pointer;">
                    &#128065; Cancella
                </button>
                <button onclick="clearEnlargedDrawing()" style="padding: 8px 15px; background: #6a4a4a; border: none; color: white; border-radius: 4px; cursor: pointer;">
                    &#128465; Reset
                </button>
                <span style="color: #888;">|</span>
                <label style="color: #888; display: flex; align-items: center; gap: 5px;">
                    Pennello: <span id="enlargedBrushSizeValue">20</span>px
                    <input type="range" id="enlargedBrushSize" min="5" max="80" value="20" style="width: 100px;">
                </label>
            </div>

            <div id="enlargedDrawingContainer" style="position: relative; display: inline-block; border: 2px solid #555; border-radius: 8px; overflow: hidden; max-width: 100%; max-height: 70vh;">
                <img id="enlargedDrawingBaseImage" style="max-width: 800px; max-height: 65vh; display: block;">
                <canvas id="enlargedDrawingCanvas" style="position: absolute; top: 0; left: 0; cursor: crosshair;"></canvas>
            </div>

            <div style="display: flex; gap: 10px; margin-top: 15px;">
                <button onclick="applyEnlargedDrawing()" style="flex: 1; padding: 12px; background: #4a6a4a; border: none; color: white; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 1.1em;">
                    &#10004; Applica Disegno (+ contorno auto)
                </button>
                <button onclick="closeEnlargedDrawing()" style="padding: 12px 20px; background: #4a4a6a; border: none; color: white; border-radius: 4px; cursor: pointer;">
                    Annulla
                </button>
            </div>
        </div>
    </div>

    <script>
        const isAdmin = {'true' if is_admin else 'false'};
        let data = [];
        let filteredData = [];
        let currentIndex = 0;
        let config = {{}};
        let vocabulary = {{}};
        let activeCollection = 'all';
        let selectMode = false;
        let selectedItems = new Set();

        // Load all data
        Promise.all([
            fetch('/api/config').then(r => r.json()),
            fetch('/api/data').then(r => r.json()),
            fetch('/api/v1/vocabulary').then(r => r.json()),
            fetch('/api/v1/periods').then(r => r.json())
        ]).then(([cfg, d, vocab, periods]) => {{
            config = cfg;
            data = d;
            vocabulary = vocab;
            initializeFilters(vocab, periods.periods);
            initializeViewer();
        }}).catch(err => {{
            document.getElementById('itemList').innerHTML = '<div class="loading"><p>Error: ' + err + '</p></div>';
        }});

        function initializeFilters(vocab, periods) {{
            // Populate period filter
            const periodSelect = document.getElementById('filterPeriod');
            if (periodSelect) {{
                periods.forEach(p => {{
                    periodSelect.innerHTML += `<option value="${{p}}">${{p.substring(0,40)}}</option>`;
                }});
            }}

            // Map field names to filter element IDs
            const filterMap = {{
                'decoration': 'filterDecoration',
                'vessel_type': 'filterVesselType',
                'part_type': 'filterPartType'
            }};

            // Populate other filters from vocabulary
            Object.entries(filterMap).forEach(([field, filterId]) => {{
                const select = document.getElementById(filterId);
                if (select && vocab[field]) {{
                    vocab[field].forEach(v => {{
                        select.innerHTML += `<option value="${{v.value}}">${{v.value}}</option>`;
                    }});
                }}
            }});
        }}

        function initializeViewer() {{
            const tabs = document.getElementById('collectionTabs');
            tabs.innerHTML = '<button class="collection-tab active" data-collection="all">All</button>';

            for (const [key, col] of Object.entries(config.collections || {{}})) {{
                tabs.innerHTML += `<button class="collection-tab" data-collection="${{key}}" style="--tab-color: ${{col.color}}">${{col.name}}</button>`;
            }}

            tabs.querySelectorAll('.collection-tab').forEach(tab => {{
                tab.addEventListener('click', () => {{
                    tabs.querySelectorAll('.collection-tab').forEach(t => t.classList.remove('active'));
                    tab.classList.add('active');
                    activeCollection = tab.dataset.collection;
                    applyFilters();
                }});
            }});

            ['filterMacroPeriod', 'filterPeriod', 'filterDecoration', 'filterVesselType', 'filterPartType'].forEach(id => {{
                const el = document.getElementById(id);
                if (el) el.addEventListener('change', applyFilters);
            }});
            document.getElementById('searchInput').addEventListener('input', applyFilters);

            // Setup autocomplete for edit fields
            if (isAdmin) {{
                setupAutocomplete('editDecoration', 'decorationList', 'decoration');
                setupAutocomplete('editVesselType', 'vesselTypeList', 'vessel_type');
                setupAutocomplete('editPartType', 'partTypeList', 'part_type');
                setupAutocomplete('editPeriod', 'periodList', 'period');
                setupAutocomplete('batchDecoration', 'batchDecorationList', 'decoration');
                setupAutocomplete('batchVesselType', 'batchVesselTypeList', 'vessel_type');
                setupAutocomplete('batchPartType', 'batchPartTypeList', 'part_type');
            }}

            applyFilters();
        }}

        function setupAutocomplete(inputId, listId, field) {{
            const input = document.getElementById(inputId);
            const list = document.getElementById(listId);
            if (!input || !list) return;

            input.addEventListener('focus', () => showAutocomplete(input, list, field));
            input.addEventListener('input', () => showAutocomplete(input, list, field));
            input.addEventListener('blur', () => setTimeout(() => list.classList.remove('show'), 200));
        }}

        function showAutocomplete(input, list, field) {{
            const items = vocabulary[field] || [];
            const filter = input.value.toLowerCase();
            const filtered = items.filter(i => i.value.toLowerCase().includes(filter));

            list.innerHTML = filtered.slice(0, 10).map(i =>
                `<div class="autocomplete-item" onclick="selectAutocomplete('${{input.id}}', '${{i.value}}')">${{i.value}}<span class="count">(${{i.count}})</span></div>`
            ).join('');

            list.classList.add('show');
        }}

        function selectAutocomplete(inputId, value) {{
            document.getElementById(inputId).value = value;
        }}

        function getCollectionColor(collection) {{
            return config.collections?.[collection]?.color || '#4fc3f7';
        }}

        function applyFilters() {{
            const macroPeriod = document.getElementById('filterMacroPeriod').value;
            const period = document.getElementById('filterPeriod').value;
            const decoration = document.getElementById('filterDecoration').value;
            const vesselType = document.getElementById('filterVesselType').value;
            const partType = document.getElementById('filterPartType').value;
            const search = document.getElementById('searchInput').value.toLowerCase();

            filteredData = data.filter(item => {{
                if (activeCollection !== 'all' && item.collection !== activeCollection) return false;
                if (macroPeriod && item.macro_period !== macroPeriod) return false;
                if (period && item.period !== period) return false;
                if (decoration && item.decoration !== decoration) return false;
                if (vesselType && item.vessel_type !== vesselType) return false;
                if (partType && item.part_type !== partType) return false;
                if (search && !JSON.stringify(item).toLowerCase().includes(search)) return false;
                return true;
            }});

            document.getElementById('stats').textContent = `${{filteredData.length}} / ${{data.length}}`;
            renderList();
            if (filteredData.length > 0) selectItem(0);
            else {{
                document.getElementById('imageViewer').innerHTML = '<div class="no-image"><p>No items found</p></div>';
                document.getElementById('metadataGrid').innerHTML = '';
            }}
        }}

        function renderList() {{
            const list = document.getElementById('itemList');
            if (filteredData.length === 0) {{
                list.innerHTML = '<div class="no-image"><p>No items</p></div>';
                return;
            }}

            list.innerHTML = filteredData.map((item, i) => `
                <div class="item-card ${{i === currentIndex ? 'active' : ''}} ${{selectedItems.has(item.id) ? 'selected' : ''}}"
                     onclick="handleCardClick(event, ${{i}})"
                     data-index="${{i}}"
                     style="--collection-color: ${{getCollectionColor(item.collection)}}">
                    <input type="checkbox" class="item-checkbox"
                           ${{selectedItems.has(item.id) ? 'checked' : ''}}
                           onclick="toggleItemSelection(event, '${{item.id}}')">
                    <div class="thumb-placeholder" data-src="${{item.image_path || ''}}"></div>
                    <div class="info">
                        <div class="id">${{item.id || 'N/A'}}</div>
                        <div class="period">${{(item.period || '').substring(0, 30)}}</div>
                        <div class="tags">
                            <span class="tag ${{item.decoration || ''}}">${{item.decoration || '?'}}</span>
                            <span class="tag vessel">${{item.vessel_type || '?'}}</span>
                        </div>
                    </div>
                </div>
            `).join('');

            if (selectMode) list.classList.add('select-mode');
            else list.classList.remove('select-mode');

            // Lazy load thumbnails
            lazyLoadThumbnails();
        }}

        // Lazy loading with Intersection Observer
        let observer = null;
        function lazyLoadThumbnails() {{
            if (observer) observer.disconnect();

            observer = new IntersectionObserver((entries) => {{
                entries.forEach(entry => {{
                    if (entry.isIntersecting) {{
                        const placeholder = entry.target;
                        const src = placeholder.dataset.src;
                        if (src) {{
                            const img = document.createElement('img');
                            img.src = src;
                            img.loading = 'lazy';
                            img.onerror = () => img.style.display = 'none';
                            img.onload = () => placeholder.replaceWith(img);
                        }}
                        observer.unobserve(placeholder);
                    }}
                }});
            }}, {{ rootMargin: '100px' }});

            document.querySelectorAll('.thumb-placeholder').forEach(el => observer.observe(el));
        }}

        function handleCardClick(event, index) {{
            if (selectMode && event.target.type !== 'checkbox') {{
                toggleItemSelection(event, filteredData[index].id);
            }} else if (event.target.type !== 'checkbox') {{
                selectItem(index);
            }}
        }}

        function toggleSelectMode() {{
            selectMode = !selectMode;
            const btn = document.getElementById('selectBtn');
            const list = document.getElementById('itemList');

            if (selectMode) {{
                btn.classList.add('active');
                btn.innerHTML = '&#10003; Exit Sel.';
                list.classList.add('select-mode');
            }} else {{
                btn.classList.remove('active');
                btn.innerHTML = '&#9745; Select';
                list.classList.remove('select-mode');
                selectedItems.clear();
            }}
            updateSelectionUI();
            renderList();
        }}

        function toggleItemSelection(event, itemId) {{
            event.stopPropagation();
            if (selectedItems.has(itemId)) selectedItems.delete(itemId);
            else selectedItems.add(itemId);
            updateSelectionUI();
            renderList();
        }}

        function updateSelectionUI() {{
            const count = selectedItems.size;
            document.getElementById('selectionCount').textContent = `${{count}} sel.`;
            document.getElementById('selectionCount').classList.toggle('visible', count > 0);
            const batchDelete = document.getElementById('batchDeleteBtn');
            const batchEdit = document.getElementById('batchEditBtn');
            if (batchDelete) batchDelete.classList.toggle('visible', count > 0);
            if (batchEdit) batchEdit.classList.toggle('visible', count > 0);
        }}

        function selectItem(index) {{
            currentIndex = index;
            const item = filteredData[index];
            const color = getCollectionColor(item.collection);

            document.querySelectorAll('.item-card').forEach((el, i) => el.classList.toggle('active', i === index));
            document.querySelector('.item-card.active')?.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});

            const viewer = document.getElementById('imageViewer');
            const cacheBuster = item.rotated ? '?t=' + Date.now() : '';
            viewer.innerHTML = `
                <div class="image-spinner"></div>
                <img class="loading" id="mainImage" src="${{item.image_path}}${{cacheBuster}}"
                     onload="this.classList.remove('loading'); this.previousElementSibling.style.display='none'; initMeasureTool();"
                     onerror="this.outerHTML='<div class=\\'no-image\\'><p>Image not found</p></div>'; document.querySelector('.image-spinner')?.remove();">
                <button class="nav-btn prev" onclick="navigate(-1)">&#8249;</button>
                <button class="nav-btn next" onclick="navigate(1)">&#8250;</button>
                {rotate_buttons}
                <div class="measure-toolbar" id="measureToolbar">
                    <button class="measure-btn calibrate" id="calibrateBtn" onclick="startCalibration()" title="Calibrate scale">&#128207; Calibrate</button>
                    <button class="measure-btn" id="measureBtn" onclick="toggleMeasure()" title="Measure distance">&#128207; Measure</button>
                    <button class="measure-btn" id="clearMeasureBtn" onclick="clearMeasurements()" title="Clear all">&#10006; Clear</button>
                    <button class="measure-btn" id="saveMeasureBtn" onclick="saveMeasurements()" title="Save measurements" style="background:rgba(33,150,243,0.7);">&#128190; Save</button>
                    <button class="measure-btn" id="autoScaleBtn" onclick="autoDetectScale()" title="Auto-detect scale from PDF" style="background:rgba(156,39,176,0.7);">&#128269; Auto Scale</button>
                    <button class="measure-btn" id="view3dBtn" onclick="open3DViewer()" title="View 3D reconstruction" style="background:rgba(76,175,80,0.7);">&#127912; 3D</button>
                </div>
                <canvas class="measure-canvas" id="measureCanvas"></canvas>
                <div class="measure-info" id="measureInfo" style="display:none;"></div>
            `;
            // Reset measurement state when changing images
            resetMeasurementState();

            const fields = [
                {{ key: 'id', label: 'ID' }},
                {{ key: 'collection', label: 'Collection' }},
                {{ key: 'decoration', label: 'Decoration', class: 'decoration' }},
                {{ key: 'motivo_decorativo', label: 'Decorative Motif' }},
                {{ key: 'sintassi_decorativa', label: 'Decorative Syntax' }},
                {{ key: 'vessel_type', label: 'Vessel Type', class: 'vessel_type' }},
                {{ key: 'part_type', label: 'Part' }},
                {{ key: 'macro_period', label: 'Macro-Period' }},
                {{ key: 'period', label: 'Period', class: 'period' }},
                {{ key: 'scala_metrica', label: 'Scale' }},
                {{ key: 'larghezza_cm', label: 'Width (cm)' }},
                {{ key: 'altezza_cm', label: 'Height (cm)' }},
                {{ key: 'page_ref', label: 'PDF Ref', class: 'page-ref', clickable: true }},
                {{ key: 'figure_num', label: 'Figure' }},
                {{ key: 'pottery_id', label: 'Pottery ID' }},
            ];

            document.getElementById('metadataGrid').innerHTML = fields
                .filter(f => item[f.key])
                .map(f => `
                    <div class="meta-item ${{f.class || ''}}" style="--collection-color: ${{color}}">
                        <div class="label">${{f.label}}</div>
                        <div class="value" ${{f.clickable ? `onclick="openPdfAtPage('${{item[f.key]}}', '${{item.collection}}')"` : ''}}>
                            ${{item[f.key] || '-'}}
                        </div>
                    </div>
                `).join('');

            // Preload adjacent images for smooth navigation
            preloadAdjacent(index);
        }}

        function navigate(dir) {{
            if (filteredData.length === 0) return;
            let idx = currentIndex + dir;
            if (idx < 0) idx = filteredData.length - 1;
            if (idx >= filteredData.length) idx = 0;
            selectItem(idx);
        }}

        // Preload adjacent images for smooth navigation
        function preloadAdjacent(index) {{
            const preloadIndexes = [index - 1, index + 1, index - 2, index + 2];
            preloadIndexes.forEach(i => {{
                if (i >= 0 && i < filteredData.length && filteredData[i]?.image_path) {{
                    const img = new Image();
                    img.src = filteredData[i].image_path;
                }}
            }});
        }}

        function rotateImage(degrees) {{
            if (!isAdmin) return;
            const item = filteredData[currentIndex];
            if (!item) return;

            fetch('/api/rotate-image', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ path: item.image_path, degrees: degrees }})
            }})
            .then(r => r.json())
            .then(result => {{
                if (result.success) {{
                    item.rotated = true;  // Mark for cache-busting
                    selectItem(currentIndex);
                }} else {{
                    alert('Rotation error: ' + (result.error || 'Unknown'));
                }}
            }})
            .catch(err => alert('Error: ' + err));
        }}

        function flipImage(direction) {{
            if (!isAdmin) return;
            const item = filteredData[currentIndex];
            if (!item) return;

            fetch('/api/flip-image', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ path: item.image_path, direction: direction }})
            }})
            .then(r => r.json())
            .then(result => {{
                if (result.success) {{
                    item.rotated = true;  // Mark for cache-busting
                    selectItem(currentIndex);
                }} else {{
                    alert('Flip error: ' + (result.error || 'Unknown'));
                }}
            }})
            .catch(err => alert('Error: ' + err));
        }}

        // PDF Viewer with PDF.js
        let pdfDoc = null;
        let pdfPageNum = 1;
        let pdfScale = 1.5;
        let pdfPath = '';
        let pdfSearchText = '';

        // Initialize PDF.js worker
        if (typeof pdfjsLib !== 'undefined') {{
            pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
        }}

        function openPdfAtPage(pageRef, collection) {{
            const item = filteredData[currentIndex];
            const page = pageRef || item?.page_ref || '1';
            const coll = collection || item?.collection || '';

            // Extract page number from reference
            const pageMatch = page.match(/p+\\.?\\s*(\\d+)/i);
            const pageNum = parseInt(pageMatch ? pageMatch[1] : '1');

            // Get search text for highlighting (figure number or pottery ID)
            pdfSearchText = item?.figure_num || item?.pottery_id || item?.id || '';

            // Get PDF path from config
            pdfPath = config.collections?.[coll]?.pdf;
            if (!pdfPath) {{
                alert('PDF not configured for this collection');
                return;
            }}

            // Show modal and load PDF
            document.getElementById('pdfModal').classList.add('active');
            document.getElementById('pdfFileName').textContent = pdfPath.split('/').pop();
            document.getElementById('pdfLoading').style.display = 'flex';
            document.getElementById('pdfCanvasWrapper').style.display = 'none';

            loadPdf('/' + pdfPath, pageNum);
        }}

        async function loadPdf(url, targetPage) {{
            try {{
                pdfDoc = await pdfjsLib.getDocument(url).promise;
                pdfPageNum = Math.min(Math.max(1, targetPage), pdfDoc.numPages);
                document.getElementById('pdfTotalPages').textContent = pdfDoc.numPages;
                await renderPdfPage();
            }} catch (err) {{
                console.error('PDF load error:', err);
                document.getElementById('pdfLoading').innerHTML =
                    '<span style="color:#f44336;">Error loading PDF: ' + err.message + '</span>';
            }}
        }}

        async function renderPdfPage() {{
            if (!pdfDoc) return;

            const page = await pdfDoc.getPage(pdfPageNum);
            const viewport = page.getViewport({{ scale: pdfScale }});
            const canvas = document.getElementById('pdfCanvas');
            const ctx = canvas.getContext('2d');

            canvas.width = viewport.width;
            canvas.height = viewport.height;

            await page.render({{
                canvasContext: ctx,
                viewport: viewport
            }}).promise;

            document.getElementById('pdfCurrentPage').textContent = pdfPageNum;
            document.getElementById('pdfLoading').style.display = 'none';
            document.getElementById('pdfCanvasWrapper').style.display = 'block';

            // Update navigation buttons
            document.getElementById('pdfPrevBtn').disabled = (pdfPageNum <= 1);
            document.getElementById('pdfNextBtn').disabled = (pdfPageNum >= pdfDoc.numPages);

            // Try to find and highlight text
            if (pdfSearchText) {{
                highlightTextOnPage(page, viewport);
            }}
        }}

        async function highlightTextOnPage(page, viewport) {{
            try {{
                const textContent = await page.getTextContent();
                const wrapper = document.getElementById('pdfCanvasWrapper');

                // Remove old highlights
                wrapper.querySelectorAll('.highlight-overlay').forEach(el => el.remove());

                const searchLower = pdfSearchText.toLowerCase();
                textContent.items.forEach(item => {{
                    if (item.str.toLowerCase().includes(searchLower)) {{
                        const tx = pdfjsLib.Util.transform(viewport.transform, item.transform);
                        const highlight = document.createElement('div');
                        highlight.className = 'highlight-overlay';
                        highlight.style.left = tx[4] + 'px';
                        highlight.style.top = (viewport.height - tx[5] - item.height * pdfScale) + 'px';
                        highlight.style.width = (item.width * pdfScale) + 'px';
                        highlight.style.height = (item.height * pdfScale + 5) + 'px';
                        wrapper.appendChild(highlight);
                    }}
                }});
            }} catch (e) {{
                console.log('Text highlight not available:', e);
            }}
        }}

        function pdfGoToPage(delta) {{
            const newPage = pdfPageNum + delta;
            if (newPage >= 1 && newPage <= pdfDoc.numPages) {{
                pdfPageNum = newPage;
                document.getElementById('pdfLoading').style.display = 'flex';
                document.getElementById('pdfCanvasWrapper').style.display = 'none';
                renderPdfPage();
            }}
        }}

        function pdfZoom(delta) {{
            pdfScale = Math.max(0.5, Math.min(3, pdfScale + delta));
            renderPdfPage();
        }}

        function downloadPdfPage() {{
            const canvas = document.getElementById('pdfCanvas');
            const link = document.createElement('a');
            link.download = `page_${{pdfPageNum}}_${{pdfSearchText || 'export'}}.png`;
            link.href = canvas.toDataURL('image/png');
            link.click();
        }}

        function openFullPdf() {{
            if (pdfPath) {{
                window.open('/' + pdfPath + '#page=' + pdfPageNum, '_blank');
            }}
        }}

        function closePdfModal() {{
            document.getElementById('pdfModal').classList.remove('active');
            pdfDoc = null;
        }}

        // Close PDF modal on Escape
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape' && document.getElementById('pdfModal').classList.contains('active')) {{
                closePdfModal();
            }}
            if (e.key === 'Escape' && document.getElementById('mlModal').classList.contains('active')) {{
                closeMlModal();
            }}
        }});

        // ============ ML SIMILARITY SEARCH FUNCTIONS ============
        let mlImageData = null;
        let mlSimilarityResult = null;
        let mlChart = null;
        let allDbImages = [];
        let realPhotoMode = false;
        let manualDrawingData = null;
        let preprocessedImageData = null;  // Stores preprocessed image with current parameters
        let decoDrawMode = 'draw';
        let isDecoDrawing = false;
        let decoDrawingCtx = null;

        // Preprocessing parameters
        let preprocessParams = {{
            threshold: 18,
            minArea: 0.05,
            blur: 9
        }};

        // Toggle real photo preprocessing mode
        async function toggleRealPhotoMode(enabled) {{
            realPhotoMode = enabled;
            const optionsDiv = document.getElementById('realPhotoOptions');

            if (enabled && mlImageData) {{
                optionsDiv.style.display = 'block';
                initDrawingCanvas();
                await applyPreprocessParams();
            }} else {{
                optionsDiv.style.display = 'none';
                manualDrawingData = null;
            }}
        }}

        // Update parameter display values
        function updatePreprocessParams() {{
            const threshold = document.getElementById('contrastThreshold').value;
            const minArea = document.getElementById('minAreaSlider').value;
            const blur = document.getElementById('blurSlider').value;

            document.getElementById('thresholdValue').textContent = threshold;
            document.getElementById('minAreaValue').textContent = minArea;
            document.getElementById('blurValue').textContent = blur;

            preprocessParams = {{
                threshold: parseInt(threshold),
                minArea: parseFloat(minArea),
                blur: parseInt(blur)
            }};
        }}

        // Apply preprocessing with current parameters
        async function applyPreprocessParams() {{
            if (!mlImageData) return;

            try {{
                const response = await fetch('/api/ml/preprocess', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        image: mlImageData,
                        params: preprocessParams
                    }})
                }});
                const result = await response.json();
                if (result.success) {{
                    preprocessedImageData = result.processed_image;  // Store for later use
                    document.getElementById('preprocessedPreview').src = result.processed_image;
                }}
            }} catch (e) {{
                console.error('Preprocessing failed:', e);
            }}
        }}

        // Initialize drawing canvas for decoration
        function initDrawingCanvas() {{
            if (!mlImageData) return;

            const baseImg = document.getElementById('drawingBaseImage');
            const canvas = document.getElementById('drawingCanvas');

            baseImg.src = mlImageData;
            baseImg.onload = function() {{
                canvas.width = baseImg.offsetWidth;
                canvas.height = baseImg.offsetHeight;
                decoDrawingCtx = canvas.getContext('2d');
                decoDrawingCtx.lineCap = 'round';
                decoDrawingCtx.lineJoin = 'round';
                clearDrawing();
                setupDecoDrawingEvents(canvas);
            }};
        }}

        // Setup drawing events for decoration canvas
        function setupDecoDrawingEvents(canvas) {{
            canvas.onmousedown = function(e) {{
                isDecoDrawing = true;
                if (decoDrawingCtx) decoDrawingCtx.beginPath();
                decoDraw(e);
            }};
            canvas.onmousemove = function(e) {{
                if (isDecoDrawing) decoDraw(e);
            }};
            canvas.onmouseup = function() {{
                isDecoDrawing = false;
                if (decoDrawingCtx) decoDrawingCtx.beginPath();
            }};
            canvas.onmouseleave = function() {{
                isDecoDrawing = false;
                if (decoDrawingCtx) decoDrawingCtx.beginPath();
            }};

            // Touch support
            canvas.ontouchstart = function(e) {{
                e.preventDefault();
                isDecoDrawing = true;
                if (decoDrawingCtx) decoDrawingCtx.beginPath();
                decoDrawTouch(e);
            }};
            canvas.ontouchmove = function(e) {{
                e.preventDefault();
                if (isDecoDrawing) decoDrawTouch(e);
            }};
            canvas.ontouchend = function() {{
                isDecoDrawing = false;
                if (decoDrawingCtx) decoDrawingCtx.beginPath();
            }};
        }}

        function decoDraw(e) {{
            if (!decoDrawingCtx) return;
            const canvas = e.target;
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const brushSize = document.getElementById('brushSize').value;
            document.getElementById('brushSizeValue').textContent = brushSize;

            decoDrawingCtx.lineWidth = brushSize;
            if (decoDrawMode === 'draw') {{
                decoDrawingCtx.globalCompositeOperation = 'source-over';
                decoDrawingCtx.strokeStyle = 'rgba(0, 0, 0, 0.8)';
            }} else {{
                decoDrawingCtx.globalCompositeOperation = 'destination-out';
                decoDrawingCtx.strokeStyle = 'rgba(255, 255, 255, 1)';
            }}

            decoDrawingCtx.lineTo(x, y);
            decoDrawingCtx.stroke();
            decoDrawingCtx.beginPath();
            decoDrawingCtx.moveTo(x, y);
        }}

        function decoDrawTouch(e) {{
            if (!decoDrawingCtx) return;
            const canvas = e.target;
            const rect = canvas.getBoundingClientRect();
            const touch = e.touches[0];
            const x = touch.clientX - rect.left;
            const y = touch.clientY - rect.top;

            const brushSize = document.getElementById('brushSize').value;
            decoDrawingCtx.lineWidth = brushSize;
            if (decoDrawMode === 'draw') {{
                decoDrawingCtx.globalCompositeOperation = 'source-over';
                decoDrawingCtx.strokeStyle = 'rgba(0, 0, 0, 0.8)';
            }} else {{
                decoDrawingCtx.globalCompositeOperation = 'destination-out';
            }}

            decoDrawingCtx.lineTo(x, y);
            decoDrawingCtx.stroke();
            decoDrawingCtx.beginPath();
            decoDrawingCtx.moveTo(x, y);
        }}

        function setDrawMode(mode) {{
            decoDrawMode = mode;
            document.getElementById('drawModeBtn').style.background = mode === 'draw' ? '#6a4a9a' : '#4a4a6a';
            document.getElementById('eraseModeBtn').style.background = mode === 'erase' ? '#6a4a9a' : '#4a4a6a';
            if (decoDrawingCtx) decoDrawingCtx.beginPath();
        }}

        function clearDrawing() {{
            const canvas = document.getElementById('drawingCanvas');
            if (!canvas) return;

            // Get or create context
            if (!decoDrawingCtx) {{
                decoDrawingCtx = canvas.getContext('2d');
            }}

            if (decoDrawingCtx && canvas.width > 0) {{
                decoDrawingCtx.clearRect(0, 0, canvas.width, canvas.height);
                decoDrawingCtx.beginPath();
            }}
            manualDrawingData = null;
        }}

        // Apply manual drawing - sends to server to add ceramic contour
        async function applyManualDrawing() {{
            const canvas = document.getElementById('drawingCanvas');
            const baseImg = document.getElementById('drawingBaseImage');

            if (!canvas || canvas.width === 0) {{
                alert('Nessun disegno da applicare');
                return;
            }}

            // Create drawing data URL
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = baseImg.naturalWidth;
            tempCanvas.height = baseImg.naturalHeight;
            const tempCtx = tempCanvas.getContext('2d');

            // White background
            tempCtx.fillStyle = 'white';
            tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);

            // Scale and draw the user's drawing
            const scaleX = baseImg.naturalWidth / canvas.width;
            const scaleY = baseImg.naturalHeight / canvas.height;
            tempCtx.scale(scaleX, scaleY);
            tempCtx.drawImage(canvas, 0, 0);

            const drawingData = tempCanvas.toDataURL('image/png');

            // Send to server to combine with automatic contour extraction
            // Use preprocessed image if available (user changed parameters)
            try {{
                const response = await fetch('/api/ml/combine-drawing', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        original_image: mlImageData,
                        drawing: drawingData,
                        preprocessed_image: preprocessedImageData  // Include if user changed params
                    }})
                }});
                const result = await response.json();
                if (result.success) {{
                    manualDrawingData = result.combined_image;
                    document.getElementById('preprocessedPreview').src = manualDrawingData;
                    realPhotoMode = true;
                    alert('Disegno + contorno applicati! Ora puoi usare Find Similar o Classify.');
                }} else {{
                    alert('Errore: ' + (result.error || 'Unknown error'));
                }}
            }} catch (e) {{
                console.error('Error combining drawing:', e);
                // Fallback: use drawing without contour
                manualDrawingData = drawingData;
                document.getElementById('preprocessedPreview').src = manualDrawingData;
                realPhotoMode = true;
                alert('Disegno applicato (senza contorno automatico)');
            }}
        }}

        // ============ ENLARGED DRAWING MODAL ============
        let enlargedDrawingCtx = null;
        let isEnlargedDrawing = false;
        let enlargedDrawMode = 'draw';

        function openEnlargedDrawing() {{
            if (!mlImageData) {{
                alert("Carica prima un'immagine");
                return;
            }}

            document.getElementById('enlargedDrawingModal').classList.add('active');

            const baseImg = document.getElementById('enlargedDrawingBaseImage');
            const canvas = document.getElementById('enlargedDrawingCanvas');

            baseImg.src = mlImageData;
            baseImg.onload = function() {{
                canvas.width = baseImg.offsetWidth;
                canvas.height = baseImg.offsetHeight;
                enlargedDrawingCtx = canvas.getContext('2d');
                enlargedDrawingCtx.lineCap = 'round';
                enlargedDrawingCtx.lineJoin = 'round';

                // Copy existing drawing from small canvas if any
                const smallCanvas = document.getElementById('drawingCanvas');
                if (smallCanvas && smallCanvas.width > 0 && decoDrawingCtx) {{
                    const scaleX = canvas.width / smallCanvas.width;
                    const scaleY = canvas.height / smallCanvas.height;
                    enlargedDrawingCtx.scale(scaleX, scaleY);
                    enlargedDrawingCtx.drawImage(smallCanvas, 0, 0);
                    enlargedDrawingCtx.setTransform(1, 0, 0, 1, 0, 0);
                }}

                setupEnlargedDrawingEvents(canvas);
            }};
        }}

        function closeEnlargedDrawing() {{
            document.getElementById('enlargedDrawingModal').classList.remove('active');
        }}

        function setupEnlargedDrawingEvents(canvas) {{
            canvas.onmousedown = function(e) {{
                isEnlargedDrawing = true;
                if (enlargedDrawingCtx) enlargedDrawingCtx.beginPath();
                enlargedDraw(e);
            }};
            canvas.onmousemove = function(e) {{
                if (isEnlargedDrawing) enlargedDraw(e);
            }};
            canvas.onmouseup = function() {{
                isEnlargedDrawing = false;
                if (enlargedDrawingCtx) enlargedDrawingCtx.beginPath();
            }};
            canvas.onmouseleave = function() {{
                isEnlargedDrawing = false;
                if (enlargedDrawingCtx) enlargedDrawingCtx.beginPath();
            }};

            // Touch support
            canvas.ontouchstart = function(e) {{
                e.preventDefault();
                isEnlargedDrawing = true;
                if (enlargedDrawingCtx) enlargedDrawingCtx.beginPath();
                enlargedDrawTouch(e);
            }};
            canvas.ontouchmove = function(e) {{
                e.preventDefault();
                if (isEnlargedDrawing) enlargedDrawTouch(e);
            }};
            canvas.ontouchend = function() {{
                isEnlargedDrawing = false;
                if (enlargedDrawingCtx) enlargedDrawingCtx.beginPath();
            }};
        }}

        function enlargedDraw(e) {{
            if (!enlargedDrawingCtx) return;
            const canvas = e.target;
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const brushSize = document.getElementById('enlargedBrushSize').value;
            document.getElementById('enlargedBrushSizeValue').textContent = brushSize;

            enlargedDrawingCtx.lineWidth = brushSize;
            if (enlargedDrawMode === 'draw') {{
                enlargedDrawingCtx.globalCompositeOperation = 'source-over';
                enlargedDrawingCtx.strokeStyle = 'rgba(0, 0, 0, 0.8)';
            }} else {{
                enlargedDrawingCtx.globalCompositeOperation = 'destination-out';
            }}

            enlargedDrawingCtx.lineTo(x, y);
            enlargedDrawingCtx.stroke();
            enlargedDrawingCtx.beginPath();
            enlargedDrawingCtx.moveTo(x, y);
        }}

        function enlargedDrawTouch(e) {{
            if (!enlargedDrawingCtx) return;
            const canvas = e.target;
            const rect = canvas.getBoundingClientRect();
            const touch = e.touches[0];
            const x = touch.clientX - rect.left;
            const y = touch.clientY - rect.top;

            const brushSize = document.getElementById('enlargedBrushSize').value;
            enlargedDrawingCtx.lineWidth = brushSize;
            if (enlargedDrawMode === 'draw') {{
                enlargedDrawingCtx.globalCompositeOperation = 'source-over';
                enlargedDrawingCtx.strokeStyle = 'rgba(0, 0, 0, 0.8)';
            }} else {{
                enlargedDrawingCtx.globalCompositeOperation = 'destination-out';
            }}

            enlargedDrawingCtx.lineTo(x, y);
            enlargedDrawingCtx.stroke();
            enlargedDrawingCtx.beginPath();
            enlargedDrawingCtx.moveTo(x, y);
        }}

        function setEnlargedDrawMode(mode) {{
            enlargedDrawMode = mode;
            document.getElementById('enlargedDrawModeBtn').style.background = mode === 'draw' ? '#6a4a9a' : '#4a4a6a';
            document.getElementById('enlargedEraseModeBtn').style.background = mode === 'erase' ? '#6a4a9a' : '#4a4a6a';
            if (enlargedDrawingCtx) enlargedDrawingCtx.beginPath();
        }}

        function clearEnlargedDrawing() {{
            const canvas = document.getElementById('enlargedDrawingCanvas');
            if (canvas && enlargedDrawingCtx) {{
                enlargedDrawingCtx.clearRect(0, 0, canvas.width, canvas.height);
                enlargedDrawingCtx.beginPath();
            }}
        }}

        async function applyEnlargedDrawing() {{
            const canvas = document.getElementById('enlargedDrawingCanvas');
            const baseImg = document.getElementById('enlargedDrawingBaseImage');

            if (!canvas || canvas.width === 0) {{
                alert('Nessun disegno da applicare');
                return;
            }}

            // Create drawing at full resolution
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = baseImg.naturalWidth;
            tempCanvas.height = baseImg.naturalHeight;
            const tempCtx = tempCanvas.getContext('2d');

            tempCtx.fillStyle = 'white';
            tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);

            const scaleX = baseImg.naturalWidth / canvas.width;
            const scaleY = baseImg.naturalHeight / canvas.height;
            tempCtx.scale(scaleX, scaleY);
            tempCtx.drawImage(canvas, 0, 0);

            const drawingData = tempCanvas.toDataURL('image/png');

            // Send to server to combine with contour
            // Use preprocessed image if available (user changed parameters)
            try {{
                const response = await fetch('/api/ml/combine-drawing', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        original_image: mlImageData,
                        drawing: drawingData,
                        preprocessed_image: preprocessedImageData  // Include if user changed params
                    }})
                }});
                const result = await response.json();
                if (result.success) {{
                    manualDrawingData = result.combined_image;
                    document.getElementById('preprocessedPreview').src = manualDrawingData;

                    // Also copy to small canvas for display
                    const smallCanvas = document.getElementById('drawingCanvas');
                    if (smallCanvas && decoDrawingCtx) {{
                        const smallScaleX = smallCanvas.width / canvas.width;
                        const smallScaleY = smallCanvas.height / canvas.height;
                        decoDrawingCtx.clearRect(0, 0, smallCanvas.width, smallCanvas.height);
                        decoDrawingCtx.scale(smallScaleX, smallScaleY);
                        decoDrawingCtx.drawImage(canvas, 0, 0);
                        decoDrawingCtx.setTransform(1, 0, 0, 1, 0, 0);
                    }}

                    realPhotoMode = true;
                    closeEnlargedDrawing();
                    alert('Disegno + contorno applicati!');
                }} else {{
                    alert('Errore: ' + (result.error || 'Unknown error'));
                }}
            }} catch (e) {{
                console.error('Error:', e);
                manualDrawingData = drawingData;
                realPhotoMode = true;
                closeEnlargedDrawing();
                alert('Disegno applicato (senza contorno)');
            }}
        }}

        function openMlClassifier() {{
            document.getElementById('mlModal').classList.add('active');
            setupMlDropZone();
            loadCarouselImages();
        }}

        function closeMlModal() {{
            document.getElementById('mlModal').classList.remove('active');
            // Reset state
            document.getElementById('mlCarouselSection').style.display = 'none';
            document.getElementById('mlAnalysis').style.display = 'none';
            document.getElementById('mlStatsSection').style.display = 'none';
        }}

        async function loadCarouselImages() {{
            if (allDbImages.length > 0) return;
            try {{
                const response = await fetch('/api/ml/all-images', {{ method: 'POST' }});
                const result = await response.json();
                allDbImages = result.images || [];
            }} catch (e) {{
                console.error('Failed to load carousel images:', e);
            }}
        }}

        function setupMlDropZone() {{
            const dropZone = document.getElementById('mlDropZone');
            if (dropZone._initialized) return;
            dropZone._initialized = true;

            dropZone.addEventListener('dragover', (e) => {{
                e.preventDefault();
                dropZone.classList.add('dragover');
            }});

            dropZone.addEventListener('dragleave', () => {{
                dropZone.classList.remove('dragover');
            }});

            dropZone.addEventListener('drop', (e) => {{
                e.preventDefault();
                dropZone.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {{
                    processImageFile(files[0]);
                }}
            }});
        }}

        function handleMlFile(event) {{
            const file = event.target.files[0];
            if (file) {{
                processImageFile(file);
            }}
        }}

        // Drawing state
        let drawTool = null;  // 'rect' or 'free'
        let drawColor = '#ff5722';
        let drawSize = 5;
        let isDrawing = false;
        let drawStartX = 0, drawStartY = 0;
        let drawCtx = null;
        let hasROI = false;
        let originalMlImageData = null;

        function processImageFile(file) {{
            if (!file.type.startsWith('image/')) {{
                alert('Please select an image file');
                return;
            }}

            const reader = new FileReader();
            reader.onload = (e) => {{
                mlImageData = e.target.result;
                originalMlImageData = e.target.result;

                const preview = document.getElementById('mlPreview');
                const container = document.getElementById('mlImageContainer');
                const canvas = document.getElementById('mlDrawCanvas');
                const tools = document.getElementById('mlDrawTools');

                preview.src = mlImageData;

                // Wait for image to load to set canvas size
                preview.onload = () => {{
                    container.style.display = 'block';
                    tools.classList.add('visible');

                    // Set canvas size to match image
                    canvas.width = preview.offsetWidth;
                    canvas.height = preview.offsetHeight;

                    // Initialize canvas context
                    drawCtx = canvas.getContext('2d');
                    clearDrawing();

                    // Setup canvas events
                    setupCanvasEvents(canvas);
                }};

                document.getElementById('mlClassifyBtn').disabled = false;
                document.getElementById('mlExplainBtn').disabled = false;

                // Hide previous results
                document.getElementById('mlAnalysis').style.display = 'none';
                document.getElementById('mlStatsSection').style.display = 'none';
                document.getElementById('mlExplanationPanel').classList.remove('visible');
                document.getElementById('mlMatchesGrid').innerHTML = '<p style="color: #666; text-align: center; grid-column: 1/-1;">Click "Find Similar" or "Classify & Explain"</p>';
            }};
            reader.readAsDataURL(file);
        }}

        function setupCanvasEvents(canvas) {{
            canvas.onmousedown = (e) => startDraw(e, canvas);
            canvas.onmousemove = (e) => draw(e, canvas);
            canvas.onmouseup = () => endDraw();
            canvas.onmouseleave = () => endDraw();

            // Touch support
            canvas.ontouchstart = (e) => {{ e.preventDefault(); startDraw(e.touches[0], canvas); }};
            canvas.ontouchmove = (e) => {{ e.preventDefault(); draw(e.touches[0], canvas); }};
            canvas.ontouchend = () => endDraw();
        }}

        function setDrawTool(tool) {{
            drawTool = (drawTool === tool) ? null : tool;

            document.getElementById('drawRectBtn').classList.toggle('active', drawTool === 'rect');
            document.getElementById('drawFreeBtn').classList.toggle('active', drawTool === 'free');

            const canvas = document.getElementById('mlDrawCanvas');
            canvas.style.cursor = drawTool ? 'crosshair' : 'default';
            canvas.style.pointerEvents = drawTool ? 'auto' : 'none';
        }}

        function setDrawColor(color) {{
            drawColor = color;
            document.querySelectorAll('.draw-color').forEach(el => {{
                el.classList.toggle('active', el.style.background === color || el.style.backgroundColor === color);
            }});
        }}

        function setDrawSize(size) {{
            drawSize = parseInt(size);
        }}

        function startDraw(e, canvas) {{
            if (!drawTool) return;

            isDrawing = true;
            const rect = canvas.getBoundingClientRect();
            drawStartX = e.clientX - rect.left;
            drawStartY = e.clientY - rect.top;

            if (drawTool === 'free') {{
                drawCtx.beginPath();
                drawCtx.moveTo(drawStartX, drawStartY);
            }}
        }}

        function draw(e, canvas) {{
            if (!isDrawing || !drawTool) return;

            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            if (drawTool === 'free') {{
                drawCtx.lineTo(x, y);
                drawCtx.strokeStyle = drawColor;
                drawCtx.lineWidth = drawSize;
                drawCtx.lineCap = 'round';
                drawCtx.lineJoin = 'round';
                drawCtx.stroke();
                hasROI = true;
                document.getElementById('roiIndicator').classList.add('visible');
            }} else if (drawTool === 'rect') {{
                // Redraw for live preview
                drawCtx.clearRect(0, 0, canvas.width, canvas.height);
                drawCtx.strokeStyle = drawColor;
                drawCtx.lineWidth = drawSize;
                drawCtx.setLineDash([5, 5]);
                drawCtx.strokeRect(drawStartX, drawStartY, x - drawStartX, y - drawStartY);
                drawCtx.setLineDash([]);
            }}
        }}

        function endDraw() {{
            if (!isDrawing) return;
            isDrawing = false;

            if (drawTool === 'rect') {{
                hasROI = true;
                document.getElementById('roiIndicator').classList.add('visible');
            }}

            // Update mlImageData with cropped/masked region if ROI exists
            if (hasROI) {{
                applyROIToImage();
            }}
        }}

        function clearDrawing() {{
            if (!drawCtx) return;

            const canvas = document.getElementById('mlDrawCanvas');
            drawCtx.clearRect(0, 0, canvas.width, canvas.height);
            hasROI = false;
            document.getElementById('roiIndicator').classList.remove('visible');

            // Restore original image
            mlImageData = originalMlImageData;
        }}

        function applyROIToImage() {{
            // Get the drawn region and create a masked/cropped version
            const canvas = document.getElementById('mlDrawCanvas');
            const preview = document.getElementById('mlPreview');

            // Create a composite image with the ROI highlighted
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = preview.naturalWidth;
            tempCanvas.height = preview.naturalHeight;
            const tempCtx = tempCanvas.getContext('2d');

            // Draw original image
            tempCtx.drawImage(preview, 0, 0);

            // Scale the drawing to match original image size
            const scaleX = preview.naturalWidth / canvas.width;
            const scaleY = preview.naturalHeight / canvas.height;

            // Get the bounding box of drawn content
            const imageData = drawCtx.getImageData(0, 0, canvas.width, canvas.height);
            let minX = canvas.width, minY = canvas.height, maxX = 0, maxY = 0;
            let hasPixels = false;

            for (let y = 0; y < canvas.height; y++) {{
                for (let x = 0; x < canvas.width; x++) {{
                    const i = (y * canvas.width + x) * 4;
                    if (imageData.data[i + 3] > 0) {{  // Alpha > 0
                        hasPixels = true;
                        minX = Math.min(minX, x);
                        minY = Math.min(minY, y);
                        maxX = Math.max(maxX, x);
                        maxY = Math.max(maxY, y);
                    }}
                }}
            }}

            if (hasPixels && (maxX - minX) > 10 && (maxY - minY) > 10) {{
                // Crop to ROI with some padding
                const padding = 10;
                const cropX = Math.max(0, (minX - padding) * scaleX);
                const cropY = Math.max(0, (minY - padding) * scaleY);
                const cropW = Math.min(preview.naturalWidth - cropX, (maxX - minX + padding * 2) * scaleX);
                const cropH = Math.min(preview.naturalHeight - cropY, (maxY - minY + padding * 2) * scaleY);

                // Create cropped canvas
                const croppedCanvas = document.createElement('canvas');
                croppedCanvas.width = cropW;
                croppedCanvas.height = cropH;
                const croppedCtx = croppedCanvas.getContext('2d');

                croppedCtx.drawImage(preview, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);

                // Update mlImageData with cropped image
                mlImageData = croppedCanvas.toDataURL('image/png');

                console.log('ROI applied: cropped to', cropW, 'x', cropH);
            }}
        }}

        function updateThreshold(value) {{
            document.getElementById('thresholdValue').textContent = value + '%';
        }}

        async function runSimilaritySearch() {{
            if (!mlImageData) return;

            const btn = document.getElementById('mlClassifyBtn');
            btn.disabled = true;
            btn.innerHTML = '&#9203; Analyzing...';

            // Show carousel section
            const carouselSection = document.getElementById('mlCarouselSection');
            carouselSection.style.display = 'block';

            // Build and animate carousel
            await buildAndAnimateCarousel();

            try {{
                const threshold = parseInt(document.getElementById('mlThreshold').value) / 100;

                // Use manual drawing if available, otherwise original image
                const imageToUse = manualDrawingData || mlImageData;
                const usePreprocess = realPhotoMode && !manualDrawingData;

                const response = await fetch('/api/ml/similar', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        image: imageToUse,
                        top_k: 30,
                        threshold: threshold,
                        preprocess: usePreprocess,
                        params: preprocessParams
                    }})
                }});

                // Check for HTTP errors (502, 500, etc.)
                if (!response.ok) {{
                    throw new Error('Server error: ' + response.status + ' - ' + response.statusText);
                }}

                const result = await response.json();

                if (result.error) {{
                    alert('Search error: ' + result.error);
                    btn.innerHTML = '&#128269; Find Similar Decorations';
                    btn.disabled = false;
                    return;
                }}

                // Validate response structure
                if (!result.similar_items) {{
                    throw new Error('Invalid response: missing similar_items');
                }}

                mlSimilarityResult = result;

                // Stop carousel animation
                document.getElementById('mlCarousel').classList.add('paused');
                document.getElementById('mlProgressBar').style.width = '100%';
                document.getElementById('mlProgressText').textContent = 'Analysis complete! Compared ' + (result.total_compared || 0) + ' images.';

                // Highlight matched items in carousel
                highlightMatchesInCarousel(result.similar_items || []);

                // Display analysis
                displayAnalysis(result.analysis || {{}});

                // Display statistics chart
                displayStatistics(result.statistics || {{}}, result.similar_items || []);

                // Display matches
                displaySimilarMatches(result.similar_items || []);

            }} catch (err) {{
                alert('Error: ' + err.message);
            }} finally {{
                btn.innerHTML = '&#128269; Find Similar';
                btn.disabled = false;
            }}
        }}

        async function runExplainClassification() {{
            if (!mlImageData) return;

            const btn = document.getElementById('mlExplainBtn');
            btn.disabled = true;
            btn.innerHTML = '&#9203; Analyzing...';

            try {{
                // Use manual drawing if available, otherwise original image
                const imageToUse = manualDrawingData || mlImageData;
                const usePreprocess = realPhotoMode && !manualDrawingData;

                const response = await fetch('/api/ml/explain', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ image: imageToUse, preprocess: usePreprocess, params: preprocessParams }})
                }});

                const result = await response.json();

                if (result.error) {{
                    alert('Error: ' + result.error);
                    return;
                }}

                // Store result for heatmap toggle
                window.lastExplainResult = result;

                // Open explain modal
                document.getElementById('explainModal').classList.add('active');

                // Display images
                document.getElementById('explainOriginal').src = mlImageData;
                document.getElementById('explainHeatmap').src = result.heatmap;

                // Display predictions in modal
                const predsDiv = document.getElementById('explainPredictions');
                const preds = result.predictions;
                predsDiv.innerHTML = `
                    <div class="explain-pred-card period">
                        <div class="card-header">
                            <span class="card-label">&#128197; Periodo</span>
                            <span class="card-conf">${{preds.period.confidence.toFixed(1)}}%</span>
                        </div>
                        <div class="card-value">${{preds.period.class || 'Non determinato'}}</div>
                        <div class="card-bar"><div class="card-bar-fill" style="width: ${{preds.period.confidence}}%"></div></div>
                    </div>
                    <div class="explain-pred-card decoration">
                        <div class="card-header">
                            <span class="card-label">&#127912; Decorazione</span>
                            <span class="card-conf">${{preds.decoration.confidence.toFixed(1)}}%</span>
                        </div>
                        <div class="card-value">${{preds.decoration.class || 'Non determinato'}}</div>
                        <div class="card-bar"><div class="card-bar-fill" style="width: ${{preds.decoration.confidence}}%"></div></div>
                    </div>
                    <div class="explain-pred-card vessel">
                        <div class="card-header">
                            <span class="card-label">&#127994; Tipo Vaso</span>
                            <span class="card-conf">${{preds.vessel_type.confidence.toFixed(1)}}%</span>
                        </div>
                        <div class="card-value">${{preds.vessel_type.class || 'Non determinato'}}</div>
                        <div class="card-bar"><div class="card-bar-fill" style="width: ${{preds.vessel_type.confidence}}%"></div></div>
                    </div>
                `;

                // Display explanations in modal
                const explDiv = document.getElementById('explainReasons');
                const expl = result.explanations;
                explDiv.innerHTML = `
                    <h3>&#128161; Perché questa classificazione?</h3>
                    <p><strong>&#128197; Periodo:</strong> ${{expl.period_reason}}</p>
                    <p><strong>&#127912; Decorazione:</strong> ${{expl.decoration_reason}}</p>
                    <p><strong>&#127994; Tipo:</strong> ${{expl.vessel_reason}}</p>
                    <div class="focus-highlight">
                        <p>&#128065; <strong>Focus visivo:</strong> ${{expl.visual_focus}}</p>
                    </div>
                    <p style="margin-top: 20px; padding: 15px; background: rgba(76, 175, 80, 0.15); border-radius: 8px; border-left: 4px solid #4caf50;">
                        <strong>&#9989; Riepilogo:</strong> ${{expl.summary}}
                    </p>
                `;

            }} catch (err) {{
                alert('Error: ' + err.message);
            }} finally {{
                btn.innerHTML = '&#129504; Classify & Explain';
                btn.disabled = false;
            }}
        }}

        function closeExplainModal() {{
            document.getElementById('explainModal').classList.remove('active');
        }}

        // Close explain modal on Escape
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape' && document.getElementById('explainModal').classList.contains('active')) {{
                closeExplainModal();
            }}
        }});

        async function buildAndAnimateCarousel() {{
            const carousel = document.getElementById('mlCarousel');
            const progressBar = document.getElementById('mlProgressBar');
            const progressText = document.getElementById('mlProgressText');

            // Use database images or data array
            const images = allDbImages.length > 0 ? allDbImages : data;

            // Build carousel HTML (duplicate for seamless loop)
            const carouselItems = images.slice(0, 100);
            carousel.innerHTML = [...carouselItems, ...carouselItems].map((item, i) => `
                <div class="ml-carousel-item" data-id="${{item.id}}" data-index="${{i}}">
                    <img src="${{item.image_path || ''}}" alt="${{item.id}}"
                         onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%2280%22 height=%22100%22><rect fill=%22%23222%22 width=%2280%22 height=%22100%22/></svg>'">
                </div>
            `).join('');

            carousel.classList.remove('paused');

            // Animate progress
            let progress = 0;
            const progressInterval = setInterval(() => {{
                progress += 2;
                if (progress > 95) {{
                    clearInterval(progressInterval);
                }}
                progressBar.style.width = progress + '%';
                progressText.textContent = 'Analyzing ' + Math.floor(progress * images.length / 100) + ' of ' + images.length + ' images...';

                // Animate "analyzing" effect on carousel items
                const items = carousel.querySelectorAll('.ml-carousel-item');
                items.forEach(item => item.classList.remove('analyzing'));
                const randomIdx = Math.floor(Math.random() * Math.min(items.length, 20));
                if (items[randomIdx]) items[randomIdx].classList.add('analyzing');
            }}, 100);

            // Store interval for cleanup
            carousel._progressInterval = progressInterval;
        }}

        function highlightMatchesInCarousel(matches) {{
            const carousel = document.getElementById('mlCarousel');
            const matchIds = new Set(matches.map(m => m.id));

            carousel.querySelectorAll('.ml-carousel-item').forEach(item => {{
                item.classList.remove('analyzing');
                if (matchIds.has(item.dataset.id)) {{
                    item.classList.add('matched');
                }}
            }});

            if (carousel._progressInterval) {{
                clearInterval(carousel._progressInterval);
            }}
        }}

        function displayAnalysis(analysis) {{
            const analysisSection = document.getElementById('mlAnalysis');
            const analysisText = document.getElementById('mlAnalysisText');
            const referencesDiv = document.getElementById('mlReferences');

            if (!analysis || !analysis.text) {{
                analysisSection.style.display = 'none';
                return;
            }}

            // Convert markdown bold to HTML
            const formattedText = analysis.text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
            analysisText.innerHTML = formattedText;

            // Build references
            if (analysis.references && analysis.references.length > 0) {{
                referencesDiv.innerHTML = `
                    <h4>&#128218; Bibliographic References</h4>
                    ${{analysis.references.map(ref => `
                        <span class="ml-ref-link" onclick="openPdfReference('${{ref.source_pdf}}', '${{ref.page_ref}}')">
                            ${{ref.id}} (${{ref.collection}}, ${{ref.page_ref}})
                        </span>
                    `).join('')}}
                `;
                referencesDiv.style.display = 'block';
            }} else {{
                referencesDiv.style.display = 'none';
            }}

            analysisSection.style.display = 'block';
        }}

        function openPdfReference(pdfPath, pageRef) {{
            // Extract page number
            const pageMatch = pageRef.match(/p+\.?\s*(\d+)/i);
            const pageNum = pageMatch ? pageMatch[1] : '1';
            openPdfAtPageDirect(pdfPath, pageNum);
        }}

        function displayStatistics(stats, items) {{
            const statsSection = document.getElementById('mlStatsSection');
            const statsGrid = document.getElementById('mlStatsGrid');

            if (!stats || !items || items.length === 0) {{
                statsSection.style.display = 'none';
                return;
            }}

            // Build stats grid
            statsGrid.innerHTML = `
                <div class="ml-stat-box">
                    <div class="stat-value">${{items.length}}</div>
                    <div class="stat-label">Similar Found</div>
                </div>
                <div class="ml-stat-box">
                    <div class="stat-value">${{stats.similarity_range.max.toFixed(0)}}%</div>
                    <div class="stat-label">Best Match</div>
                </div>
                <div class="ml-stat-box">
                    <div class="stat-value">${{stats.similarity_range.avg.toFixed(0)}}%</div>
                    <div class="stat-label">Avg Similarity</div>
                </div>
            `;

            // Build chart
            const ctx = document.getElementById('mlChart').getContext('2d');

            // Destroy old chart if exists
            if (mlChart) {{
                mlChart.destroy();
            }}

            // Period distribution data
            const periodData = stats.period_distribution || {{}};
            const periodLabels = Object.keys(periodData);
            const periodValues = Object.values(periodData);

            mlChart = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: periodLabels,
                    datasets: [{{
                        label: 'Period Distribution',
                        data: periodValues,
                        backgroundColor: [
                            'rgba(156, 39, 176, 0.7)',
                            'rgba(233, 30, 99, 0.7)',
                            'rgba(79, 195, 247, 0.7)',
                            'rgba(255, 152, 0, 0.7)'
                        ],
                        borderColor: [
                            'rgba(156, 39, 176, 1)',
                            'rgba(233, 30, 99, 1)',
                            'rgba(79, 195, 247, 1)',
                            'rgba(255, 152, 0, 1)'
                        ],
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }},
                        title: {{
                            display: true,
                            text: 'Period Distribution of Similar Ceramics',
                            color: '#aaa'
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            grid: {{ color: 'rgba(255,255,255,0.1)' }},
                            ticks: {{ color: '#aaa' }}
                        }},
                        x: {{
                            grid: {{ display: false }},
                            ticks: {{ color: '#aaa' }}
                        }}
                    }}
                }}
            }});

            statsSection.style.display = 'block';
        }}

        let heatmapsVisible = false;
        let matchHeatmaps = {{}};

        function displaySimilarMatches(items) {{
            const grid = document.getElementById('mlMatchesGrid');
            const toggleBtn = document.getElementById('heatmapToggleBtn');
            document.getElementById('mlMatchCount').textContent = '(' + items.length + ')';

            if (items.length === 0) {{
                grid.innerHTML = '<p style="color: #888; text-align: center; grid-column: 1/-1;">No visually similar ceramics found above threshold</p>';
                toggleBtn.style.display = 'none';
                return;
            }}

            // Show heatmap toggle button
            toggleBtn.style.display = 'inline-block';
            heatmapsVisible = false;
            toggleBtn.innerHTML = '&#128293; Show Heatmaps';
            toggleBtn.classList.remove('active');

            grid.innerHTML = items.map(item => `
                <div class="ml-match-card" data-id="${{item.id}}" onclick="selectMlMatch('${{item.id}}')">
                    <div class="match-img-container" style="position: relative;">
                        <img class="match-original" src="${{item.image_path || ''}}" alt="${{item.id}}"
                             onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22150%22 height=%22120%22><rect fill=%22%23333%22 width=%22150%22 height=%22120%22/><text fill=%22%23666%22 x=%2275%22 y=%2260%22 text-anchor=%22middle%22>No image</text></svg>'">
                        <img class="heatmap-overlay" src="" alt="heatmap" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0; transition: opacity 0.3s;">
                    </div>
                    <div class="match-id">${{item.id}}</div>
                    <div class="match-period">${{item.macro_period || item.period || 'N/A'}}</div>
                    <div class="match-confidence">${{item.similarity}}% similar</div>
                </div>
            `).join('');

            // Preload heatmaps for top matches (lazy load)
            matchHeatmaps = {{}};
            items.slice(0, 10).forEach(item => {{
                if (item.image_path) {{
                    generateHeatmapForMatch(item.id, item.image_path);
                }}
            }});
        }}

        async function generateHeatmapForMatch(itemId, imagePath) {{
            try {{
                // Fetch the image and convert to base64
                const response = await fetch(imagePath);
                const blob = await response.blob();
                const reader = new FileReader();

                reader.onload = async () => {{
                    const base64 = reader.result;

                    // Get heatmap from API
                    const apiResponse = await fetch('/api/ml/explain', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ image: base64 }})
                    }});

                    const result = await apiResponse.json();
                    if (result.heatmap) {{
                        matchHeatmaps[itemId] = result.heatmap;

                        // If heatmaps are visible, update this one
                        if (heatmapsVisible) {{
                            const card = document.querySelector(`.ml-match-card[data-id="${{itemId}}"] .heatmap-overlay`);
                            if (card) {{
                                card.src = result.heatmap;
                                card.style.opacity = '0.6';
                            }}
                        }}
                    }}
                }};
                reader.readAsDataURL(blob);
            }} catch (err) {{
                console.log('Error generating heatmap for', itemId, err);
            }}
        }}

        function toggleHeatmaps() {{
            heatmapsVisible = !heatmapsVisible;
            const toggleBtn = document.getElementById('heatmapToggleBtn');

            if (heatmapsVisible) {{
                toggleBtn.innerHTML = '&#128293; Hide Heatmaps';
                toggleBtn.classList.add('active');

                // Show heatmaps
                document.querySelectorAll('.ml-match-card').forEach(card => {{
                    const itemId = card.dataset.id;
                    const overlay = card.querySelector('.heatmap-overlay');
                    if (overlay && matchHeatmaps[itemId]) {{
                        overlay.src = matchHeatmaps[itemId];
                        overlay.style.opacity = '0.6';
                    }}
                }});
            }} else {{
                toggleBtn.innerHTML = '&#128293; Show Heatmaps';
                toggleBtn.classList.remove('active');

                // Hide heatmaps
                document.querySelectorAll('.heatmap-overlay').forEach(overlay => {{
                    overlay.style.opacity = '0';
                }});
            }}
        }}

        function selectMlMatch(itemId) {{
            closeMlModal();

            const index = filteredData.findIndex(item => item.id === itemId);
            if (index >= 0) {{
                selectItem(index);
            }} else {{
                const allIndex = data.findIndex(item => item.id === itemId);
                if (allIndex >= 0) {{
                    document.getElementById('filterMacroPeriod').value = '';
                    document.getElementById('filterPeriod').value = '';
                    document.getElementById('filterDecoration').value = '';
                    document.getElementById('filterVesselType').value = '';
                    document.getElementById('filterPartType').value = '';
                    document.getElementById('searchInput').value = '';
                    activeCollection = 'all';
                    document.querySelectorAll('.collection-tab').forEach(t => t.classList.remove('active'));
                    document.querySelector('.collection-tab[data-collection="all"]').classList.add('active');
                    applyFilters();

                    const newIndex = filteredData.findIndex(item => item.id === itemId);
                    if (newIndex >= 0) {{
                        selectItem(newIndex);
                    }}
                }}
            }}
        }}

        function openPdfAtPageDirect(pdfPath, pageNum) {{
            // Open PDF viewer directly
            loadPdf('/' + pdfPath, parseInt(pageNum), '');
        }}

        function openEditModal() {{
            if (!isAdmin || filteredData.length === 0) return;
            const item = filteredData[currentIndex];
            document.getElementById('editDecoration').value = item.decoration || '';
            document.getElementById('editDecorationCode').value = item.decoration_code || '';
            document.getElementById('editVesselType').value = item.vessel_type || '';
            document.getElementById('editPartType').value = item.part_type || '';
            document.getElementById('editPeriod').value = item.period || '';
            document.getElementById('editMotivoDecorativo').value = item.motivo_decorativo || '';
            document.getElementById('editSintassiDecorativa').value = item.sintassi_decorativa || '';
            document.getElementById('editScalaMetrica').value = item.scala_metrica || '';
            document.getElementById('editLarghezzaCm').value = item.larghezza_cm || '';
            document.getElementById('editAltezzaCm').value = item.altezza_cm || '';
            document.getElementById('editModal').classList.add('active');
            // Setup autocomplete for new fields
            setupAutocomplete('editMotivoDecorativo', 'motivoDecorativoList', 'motivo_decorativo');
            setupAutocomplete('editSintassiDecorativa', 'sintassiDecorativaList', 'sintassi_decorativa');
            setupAutocomplete('editScalaMetrica', 'scalaMetricaList', 'scala_metrica');
        }}

        function saveEdit() {{
            const item = filteredData[currentIndex];
            const fields = {{
                decoration: document.getElementById('editDecoration').value,
                decoration_code: document.getElementById('editDecorationCode').value,
                vessel_type: document.getElementById('editVesselType').value,
                part_type: document.getElementById('editPartType').value,
                period: document.getElementById('editPeriod').value,
                motivo_decorativo: document.getElementById('editMotivoDecorativo').value,
                sintassi_decorativa: document.getElementById('editSintassiDecorativa').value,
                scala_metrica: document.getElementById('editScalaMetrica').value,
                larghezza_cm: parseFloat(document.getElementById('editLarghezzaCm').value) || null,
                altezza_cm: parseFloat(document.getElementById('editAltezzaCm').value) || null
            }};

            fetch('/api/update-item', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ id: item.id, fields: fields }})
            }})
            .then(r => r.json())
            .then(result => {{
                if (result.success) {{
                    Object.assign(item, fields);
                    closeModal('editModal');
                    selectItem(currentIndex);
                    renderList();
                    // Update vocabulary if new terms added
                    for (const [field, value] of Object.entries(fields)) {{
                        if (value && vocabulary[field] && !vocabulary[field].find(v => v.value === value)) {{
                            vocabulary[field].push({{ value, count: 1 }});
                        }}
                    }}
                }} else {{
                    alert('Error: ' + (result.error || 'Unknown'));
                }}
            }})
            .catch(err => alert('Error: ' + err));
        }}

        function openBatchEditModal() {{
            if (!isAdmin || selectedItems.size === 0) return;
            document.getElementById('batchCount').textContent = selectedItems.size;
            document.getElementById('batchDecoration').value = '';
            document.getElementById('batchVesselType').value = '';
            document.getElementById('batchPartType').value = '';
            document.getElementById('batchEditModal').classList.add('active');
        }}

        function saveBatchEdit() {{
            const fields = {{}};
            const dec = document.getElementById('batchDecoration').value;
            const vessel = document.getElementById('batchVesselType').value;
            const part = document.getElementById('batchPartType').value;

            if (dec) fields.decoration = dec;
            if (vessel) fields.vessel_type = vessel;
            if (part) fields.part_type = part;

            if (Object.keys(fields).length === 0) {{
                alert('Select at least one field to change');
                return;
            }}

            fetch('/api/update-batch', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ ids: Array.from(selectedItems), fields: fields }})
            }})
            .then(r => r.json())
            .then(result => {{
                if (result.success) {{
                    for (const id of selectedItems) {{
                        const item = data.find(d => d.id === id);
                        if (item) Object.assign(item, fields);
                    }}
                    closeModal('batchEditModal');
                    applyFilters();
                    alert(`Updated ${{result.updated}} items`);
                }} else {{
                    alert('Error: ' + (result.error || 'Unknown'));
                }}
            }})
            .catch(err => alert('Error: ' + err));
        }}

        function confirmDelete() {{
            if (!isAdmin || filteredData.length === 0) return;
            const item = filteredData[currentIndex];
            document.getElementById('deleteMessage').innerHTML = `Delete <strong>${{item.id}}</strong>?`;
            document.getElementById('deleteModal').dataset.batch = '';
            document.getElementById('deleteModal').classList.add('active');
        }}

        function confirmBatchDelete() {{
            if (!isAdmin || selectedItems.size === 0) return;
            document.getElementById('deleteMessage').innerHTML = `Delete <strong>${{selectedItems.size}} items</strong>?`;
            document.getElementById('deleteModal').dataset.batch = 'true';
            document.getElementById('deleteModal').classList.add('active');
        }}

        function closeModal(id) {{
            document.getElementById(id).classList.remove('active');
        }}

        // ===== DECORATION CATALOG =====
        let decorationCatalog = [];
        let catalogFilter = 'all';

        async function loadDecorationCatalog() {{
            if (decorationCatalog.length > 0) return;
            try {{
                const resp = await fetch('/api/v1/decoration-catalog');
                const data = await resp.json();
                decorationCatalog = data.catalog || [];
                console.log('Loaded decoration catalog:', decorationCatalog.length, 'items');
            }} catch (e) {{
                console.error('Failed to load decoration catalog:', e);
            }}
        }}

        function openDecorationCatalog() {{
            loadDecorationCatalog().then(() => {{
                renderCatalog();
                document.getElementById('decorationCatalogModal').classList.add('active');
            }});
        }}

        function renderCatalog() {{
            const grid = document.getElementById('catalogGrid');
            const searchTerm = (document.getElementById('catalogSearch')?.value || '').toLowerCase();

            let items = decorationCatalog;

            // Filter by category
            if (catalogFilter !== 'all') {{
                items = items.filter(item => item.category === catalogFilter);
            }}

            // Filter by search term
            if (searchTerm) {{
                items = items.filter(item =>
                    item.code.toLowerCase().includes(searchTerm) ||
                    (item.name_it || '').toLowerCase().includes(searchTerm) ||
                    (item.name_en || '').toLowerCase().includes(searchTerm)
                );
            }}

            grid.innerHTML = items.map((item, idx) => `
                <div class="cat-item" data-code="${{item.code}}">
                    <div class="cat-item-thumb" onclick="selectDecorationByIndex(${{idx}})">
                        ${{item.thumbnail
                            ? `<img src="/${{item.thumbnail}}" alt="${{item.code}}" onerror="this.parentElement.innerHTML='<span style=color:#999>${{item.code}}</span>'">`
                            : `<span style="color:#999;font-size:1.5em">${{item.code}}</span>`
                        }}
                    </div>
                    <div class="cat-item-info" onclick="selectDecorationByIndex(${{idx}})">
                        <div class="cat-item-code">${{item.code}}</div>
                        <div class="cat-item-name">${{item.name_en || item.name_it || ''}}</div>
                        <div class="cat-item-category">${{getCategoryLabel(item.category)}}</div>
                        ${{item.period_diagnostic ? `<div class="cat-item-period" title="${{item.period_diagnostic}}">${{getPeriodBadge(item.period_diagnostic)}}</div>` : ''}}
                    </div>
                    ${{item.pdf_page ? `<button class="cat-pdf-btn" onclick="event.stopPropagation(); openSchmidtPdf(${{item.pdf_page}})" title="Open PDF at page ${{item.pdf_page}}">📄 p.${{item.pdf_page}}</button>` : ''}}
                </div>
            `).join('');

            // Store filtered items for selection
            window.filteredCatalogItems = items;
        }}

        function filterCatalog(category) {{
            catalogFilter = category;
            document.querySelectorAll('.cat-filter-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            renderCatalog();
        }}

        function searchCatalog(term) {{
            renderCatalog();
        }}

        function getCategoryLabel(cat) {{
            const labels = {{
                'bemalung': 'Painting',
                'kombination': 'Combination',
                'negativ': 'Incised',
                'positiv': 'Relief',
                'none': 'None'
            }};
            return labels[cat] || cat;
        }}

        function getPeriodBadge(diagnostic) {{
            if (!diagnostic) return '';
            if (diagnostic.includes('Iron Age') || diagnostic.includes('Eisenzeit')) {{
                return `<span class="cat-item-period iron-age">Iron Age</span>`;
            }} else if (diagnostic.includes('both') || diagnostic.includes('Wadi Suq')) {{
                return `<span class="cat-item-period multi-period">Multi-period</span>`;
            }} else if (diagnostic.includes('Umm an-Nar')) {{
                return `<span class="cat-item-period umm-an-nar">Umm an-Nar</span>`;
            }}
            return '';
        }}

        // Open Schmidt PDF at specific page
        function openSchmidtPdf(pageNum) {{
            console.log('Opening Schmidt PDF at page', pageNum);
            // Use browser-based PDF viewing with page fragment
            fetch(`/api/pdf-url?collection=Schmidt_Bat&page=${{pageNum}}`)
                .then(resp => resp.json())
                .then(data => {{
                    if (data.success && data.url) {{
                        window.open(data.url, '_blank');
                    }} else if (data.error) {{
                        alert('Error opening PDF: ' + data.error);
                    }}
                }})
                .catch(err => {{
                    console.error('Failed to open PDF:', err);
                }});
        }}

        function selectDecorationByIndex(idx) {{
            const items = window.filteredCatalogItems;
            if (!items || idx >= items.length) return;

            const item = items[idx];
            console.log('Selected decoration:', item);

            document.getElementById('editDecorationCode').value = item.code;

            // Also update motivo_decorativo with the English name
            const motivoInput = document.getElementById('editMotivoDecorativo');
            if (motivoInput) {{
                motivoInput.value = item.name_en || item.name_it || '';
            }}

            closeModal('decorationCatalogModal');
        }}

        function selectDecoration(code, name) {{
            document.getElementById('editDecorationCode').value = code;
            const motivoInput = document.getElementById('editMotivoDecorativo');
            if (motivoInput && name) {{
                motivoInput.value = name;
            }}
            closeModal('decorationCatalogModal');
        }}

        function executeDelete() {{
            const modal = document.getElementById('deleteModal');
            const isBatch = modal.dataset.batch === 'true';
            closeModal('deleteModal');

            if (isBatch) {{
                const items = Array.from(selectedItems).map(id => {{
                    const item = data.find(d => d.id === id);
                    return {{ id: item.id, path: item.image_path }};
                }});

                Promise.all(items.map(item =>
                    fetch(`/api/delete-image?path=${{encodeURIComponent(item.path)}}&id=${{encodeURIComponent(item.id)}}`, {{ method: 'DELETE' }})
                )).then(() => {{
                    for (const id of selectedItems) {{
                        const idx = data.findIndex(d => d.id === id);
                        if (idx > -1) data.splice(idx, 1);
                    }}
                    selectedItems.clear();
                    updateSelectionUI();
                    applyFilters();
                }});
            }} else {{
                const item = filteredData[currentIndex];
                fetch(`/api/delete-image?path=${{encodeURIComponent(item.image_path)}}&id=${{encodeURIComponent(item.id)}}`, {{ method: 'DELETE' }})
                .then(r => r.json())
                .then(result => {{
                    if (result.success) {{
                        const idx = data.findIndex(d => d.id === item.id);
                        if (idx > -1) data.splice(idx, 1);
                        applyFilters();
                    }}
                }});
            }}
        }}

        function logout() {{
            fetch('/api/logout', {{ method: 'POST' }})
                .then(() => window.location.href = '/');
        }}

        document.addEventListener('keydown', e => {{
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
            if (e.key === 'ArrowLeft') navigate(-1);
            if (e.key === 'ArrowRight') navigate(1);
            if (e.key === 'p' || e.key === 'P') openPdfAtPage();
            if (isAdmin && (e.key === 'e' || e.key === 'E')) openEditModal();
            if (isAdmin && (e.key === 'r' || e.key === 'R')) rotateImage(90);
            if (e.key === 'm' || e.key === 'M') toggleMeasure();
            if (e.key === 'Escape') {{
                closeModal('deleteModal');
                closeModal('editModal');
                closeModal('batchEditModal');
                cancelCalibration();
                if (measureMode) toggleMeasure();
            }}
        }});

        // ============================================
        // MEASUREMENT TOOL
        // ============================================
        let measureMode = false;
        let calibrationMode = false;
        let calibrationPoints = [];
        let measurePoints = [];
        let measurements = [];
        let pixelsPerCm = null;  // Will be set after calibration
        let measureCanvas = null;
        let measureCtx = null;

        function initMeasureTool() {{
            const canvas = document.getElementById('measureCanvas');
            const img = document.getElementById('mainImage');
            if (!canvas || !img) return;

            measureCanvas = canvas;
            measureCtx = canvas.getContext('2d');

            // Set canvas size to match image container
            const viewer = document.getElementById('imageViewer');
            canvas.width = viewer.clientWidth;
            canvas.height = viewer.clientHeight;

            // Add mouse events
            canvas.addEventListener('click', handleMeasureClick);
            canvas.addEventListener('mousemove', handleMeasureMove);

            // Load existing calibration for this item
            loadCalibration();
        }}

        function resetMeasurementState() {{
            measureMode = false;
            calibrationMode = false;
            calibrationPoints = [];
            measurePoints = [];
            measurements = [];
            pixelsPerCm = null;

            const canvas = document.getElementById('measureCanvas');
            if (canvas) {{
                canvas.classList.remove('active');
                const ctx = canvas.getContext('2d');
                if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
            }}

            const info = document.getElementById('measureInfo');
            if (info) info.style.display = 'none';

            const calibrateBtn = document.getElementById('calibrateBtn');
            const measureBtn = document.getElementById('measureBtn');
            if (calibrateBtn) calibrateBtn.classList.remove('active');
            if (measureBtn) measureBtn.classList.remove('active');
        }}

        function loadCalibration() {{
            const item = filteredData[currentIndex];
            if (!item) return;

            // Check if item has scala_metrica (scale info)
            if (item.scala_metrica) {{
                // Try to parse scale like "1:3" -> means 1cm drawing = 3cm real
                const match = item.scala_metrica.match(/(\\d+(?:\\.\\d+)?)\s*:\s*(\\d+(?:\\.\\d+)?)/);
                if (match) {{
                    const drawingScale = parseFloat(match[1]) / parseFloat(match[2]);
                    updateMeasureInfo('Scale from metadata: ' + item.scala_metrica + ' (use Calibrate for precise measurements)');
                }}
            }}

            // Check if we have saved calibration data
            if (item.calibration_data) {{
                try {{
                    const data = JSON.parse(item.calibration_data);
                    if (data.pixelsPerCm) {{
                        pixelsPerCm = data.pixelsPerCm;
                        updateMeasureInfo('Calibrated: ' + pixelsPerCm.toFixed(2) + ' px/cm');
                    }}
                }} catch (e) {{ }}
            }}
        }}

        function startCalibration() {{
            calibrationMode = true;
            calibrationPoints = [];
            measureMode = false;

            const canvas = document.getElementById('measureCanvas');
            if (canvas) canvas.classList.add('active');

            document.getElementById('calibrateBtn').classList.add('active');
            document.getElementById('measureBtn').classList.remove('active');

            // Show instructions in measureInfo (don't block with modal yet)
            updateMeasureInfo('&#128207; <strong>CALIBRATION:</strong> Click the <u>FIRST</u> point on a known distance (e.g., scale bar)');
        }}

        function cancelCalibration() {{
            calibrationMode = false;
            calibrationPoints = [];
            document.getElementById('calibrationModal').classList.remove('active');
            document.getElementById('calibrateBtn').classList.remove('active');

            const canvas = document.getElementById('measureCanvas');
            if (canvas && !measureMode) canvas.classList.remove('active');
            redrawMeasurements();
        }}

        function applyCalibration() {{
            const distance = parseFloat(document.getElementById('calibrationDistance').value);
            const unit = document.getElementById('calibrationUnit').value;

            if (calibrationPoints.length !== 2 || !distance || distance <= 0) {{
                alert('Invalid calibration data');
                return;
            }}

            // Convert to cm
            let distanceCm = distance;
            if (unit === 'mm') distanceCm = distance / 10;
            if (unit === 'm') distanceCm = distance * 100;

            // Calculate pixels per cm
            const dx = calibrationPoints[1].x - calibrationPoints[0].x;
            const dy = calibrationPoints[1].y - calibrationPoints[0].y;
            const pixelDistance = Math.sqrt(dx * dx + dy * dy);
            pixelsPerCm = pixelDistance / distanceCm;

            updateMeasureInfo('Calibrated: ' + pixelsPerCm.toFixed(2) + ' px/cm (' + distanceCm.toFixed(1) + ' cm reference)');

            calibrationMode = false;
            document.getElementById('calibrationModal').classList.remove('active');
            document.getElementById('calibrateBtn').classList.remove('active');

            const canvas = document.getElementById('measureCanvas');
            if (canvas && !measureMode) canvas.classList.remove('active');

            redrawMeasurements();
        }}

        function toggleMeasure() {{
            measureMode = !measureMode;
            measurePoints = [];

            const canvas = document.getElementById('measureCanvas');
            const btn = document.getElementById('measureBtn');

            if (measureMode) {{
                if (!pixelsPerCm) {{
                    updateMeasureInfo('&#9888; Not calibrated! Use Calibrate first for accurate measurements.');
                }}
                canvas.classList.add('active');
                btn.classList.add('active');
            }} else {{
                canvas.classList.remove('active');
                btn.classList.remove('active');
            }}
        }}

        function handleMeasureClick(e) {{
            if (!measureMode && !calibrationMode) return;

            const rect = measureCanvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            if (calibrationMode) {{
                calibrationPoints.push({{ x, y }});
                redrawMeasurements();

                if (calibrationPoints.length === 1) {{
                    updateMeasureInfo('&#128207; <strong>CALIBRATION:</strong> Now click the <u>SECOND</u> point');
                }} else if (calibrationPoints.length === 2) {{
                    // Show modal to enter the distance
                    document.getElementById('calibrationModal').classList.add('active');
                    document.getElementById('calibrationStep').textContent = 'Enter the real distance between the two points:';
                    document.getElementById('calibrationInput').style.display = 'block';
                    updateMeasureInfo('&#128207; Enter the known distance in the popup');
                }}
            }} else if (measureMode) {{
                measurePoints.push({{ x, y }});

                if (measurePoints.length === 2) {{
                    // Calculate and store measurement
                    const dx = measurePoints[1].x - measurePoints[0].x;
                    const dy = measurePoints[1].y - measurePoints[0].y;
                    const pixelDistance = Math.sqrt(dx * dx + dy * dy);

                    let realDistance = null;
                    let displayText = pixelDistance.toFixed(1) + ' px';

                    if (pixelsPerCm) {{
                        realDistance = pixelDistance / pixelsPerCm;
                        displayText = realDistance.toFixed(2) + ' cm (' + pixelDistance.toFixed(0) + ' px)';
                    }}

                    measurements.push({{
                        points: [...measurePoints],
                        pixelDistance: pixelDistance,
                        realDistance: realDistance,
                        displayText: displayText
                    }});

                    updateMeasureInfo('Last measurement: ' + displayText + ' | Total: ' + measurements.length + ' measurements');
                    measurePoints = [];
                    redrawMeasurements();
                }} else {{
                    redrawMeasurements();
                }}
            }}
        }}

        function handleMeasureMove(e) {{
            if (!measureMode && !calibrationMode) return;
            if ((measureMode && measurePoints.length !== 1) && (calibrationMode && calibrationPoints.length !== 1)) return;

            const rect = measureCanvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            redrawMeasurements();

            // Draw line to cursor
            measureCtx.beginPath();
            measureCtx.strokeStyle = calibrationMode ? '#ff9800' : '#4caf50';
            measureCtx.lineWidth = 2;
            measureCtx.setLineDash([5, 5]);

            const startPoint = calibrationMode ? calibrationPoints[0] : measurePoints[0];
            if (startPoint) {{
                measureCtx.moveTo(startPoint.x, startPoint.y);
                measureCtx.lineTo(x, y);
                measureCtx.stroke();

                // Show distance
                const dx = x - startPoint.x;
                const dy = y - startPoint.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                let text = dist.toFixed(0) + ' px';
                if (pixelsPerCm && measureMode) {{
                    text = (dist / pixelsPerCm).toFixed(2) + ' cm';
                }}

                measureCtx.setLineDash([]);
                measureCtx.font = '14px Arial';
                measureCtx.fillStyle = calibrationMode ? '#ff9800' : '#4caf50';
                measureCtx.fillText(text, (startPoint.x + x) / 2 + 10, (startPoint.y + y) / 2 - 10);
            }}
        }}

        function redrawMeasurements() {{
            if (!measureCtx) return;
            measureCtx.clearRect(0, 0, measureCanvas.width, measureCanvas.height);
            measureCtx.setLineDash([]);

            // Draw calibration points
            calibrationPoints.forEach((p, i) => {{
                measureCtx.beginPath();
                measureCtx.arc(p.x, p.y, 6, 0, Math.PI * 2);
                measureCtx.fillStyle = '#ff9800';
                measureCtx.fill();
                measureCtx.font = '12px Arial';
                measureCtx.fillText('C' + (i + 1), p.x + 10, p.y - 10);
            }});

            // Draw calibration line
            if (calibrationPoints.length === 2) {{
                measureCtx.beginPath();
                measureCtx.moveTo(calibrationPoints[0].x, calibrationPoints[0].y);
                measureCtx.lineTo(calibrationPoints[1].x, calibrationPoints[1].y);
                measureCtx.strokeStyle = '#ff9800';
                measureCtx.lineWidth = 2;
                measureCtx.stroke();
            }}

            // Draw saved measurements
            measurements.forEach((m, i) => {{
                measureCtx.beginPath();
                measureCtx.moveTo(m.points[0].x, m.points[0].y);
                measureCtx.lineTo(m.points[1].x, m.points[1].y);
                measureCtx.strokeStyle = '#4caf50';
                measureCtx.lineWidth = 2;
                measureCtx.stroke();

                // End points
                m.points.forEach(p => {{
                    measureCtx.beginPath();
                    measureCtx.arc(p.x, p.y, 4, 0, Math.PI * 2);
                    measureCtx.fillStyle = '#4caf50';
                    measureCtx.fill();
                }});

                // Label
                const midX = (m.points[0].x + m.points[1].x) / 2;
                const midY = (m.points[0].y + m.points[1].y) / 2;
                measureCtx.font = 'bold 12px Arial';
                measureCtx.fillStyle = '#fff';
                measureCtx.strokeStyle = '#000';
                measureCtx.lineWidth = 3;
                measureCtx.strokeText(m.displayText, midX + 5, midY - 5);
                measureCtx.fillText(m.displayText, midX + 5, midY - 5);
            }});

            // Draw current measurement point
            measurePoints.forEach((p, i) => {{
                measureCtx.beginPath();
                measureCtx.arc(p.x, p.y, 5, 0, Math.PI * 2);
                measureCtx.fillStyle = '#4caf50';
                measureCtx.fill();
            }});
        }}

        function clearMeasurements() {{
            measurements = [];
            calibrationPoints = [];
            measurePoints = [];
            redrawMeasurements();
            updateMeasureInfo('Measurements cleared');
        }}

        function updateMeasureInfo(text) {{
            const info = document.getElementById('measureInfo');
            if (info) {{
                info.innerHTML = text;
                info.style.display = 'block';
            }}
        }}

        function saveMeasurements() {{
            if (!isAdmin) {{
                alert('Only admins can save measurements');
                return;
            }}

            const item = filteredData[currentIndex];
            if (!item) return;

            // Calculate width and height from measurements if we have calibration
            let width = null, height = null;

            if (pixelsPerCm && measurements.length > 0) {{
                // Find the largest horizontal and vertical measurements
                measurements.forEach(m => {{
                    const dx = Math.abs(m.points[1].x - m.points[0].x);
                    const dy = Math.abs(m.points[1].y - m.points[0].y);

                    // If mostly horizontal (dx > dy), could be width
                    if (dx > dy && m.realDistance) {{
                        if (!width || m.realDistance > width) width = m.realDistance;
                    }}
                    // If mostly vertical, could be height
                    if (dy > dx && m.realDistance) {{
                        if (!height || m.realDistance > height) height = m.realDistance;
                    }}
                }});
            }}

            // Save calibration and measurements
            const calibrationData = JSON.stringify({{
                pixelsPerCm: pixelsPerCm,
                measurements: measurements.map(m => ({{
                    points: m.points,
                    realDistance: m.realDistance
                }}))
            }});

            fetch('/api/update-item', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    id: item.id,
                    fields: {{
                        calibration_data: calibrationData,
                        larghezza_cm: width ? parseFloat(width.toFixed(2)) : null,
                        altezza_cm: height ? parseFloat(height.toFixed(2)) : null
                    }}
                }})
            }})
            .then(r => r.json())
            .then(result => {{
                if (result.success) {{
                    updateMeasureInfo('&#10004; Measurements saved! Width: ' + (width ? width.toFixed(2) + ' cm' : 'N/A') + ', Height: ' + (height ? height.toFixed(2) + ' cm' : 'N/A'));
                    // Update local data
                    item.calibration_data = calibrationData;
                    if (width) item.larghezza_cm = width.toFixed(2);
                    if (height) item.altezza_cm = height.toFixed(2);
                    selectItem(currentIndex);  // Refresh display
                }} else {{
                    alert('Error saving: ' + (result.error || 'Unknown error'));
                }}
            }})
            .catch(err => alert('Error: ' + err));
        }}

        function autoDetectScale() {{
            const item = filteredData[currentIndex];
            if (!item) return;

            updateMeasureInfo('&#8987; Searching for scale in PDF...');

            const collection = item.collection || '';
            const pageRef = item.page_ref || '';

            fetch('/api/extract-scale?collection=' + encodeURIComponent(collection) + '&page=' + encodeURIComponent(pageRef))
                .then(r => r.json())
                .then(result => {{
                    if (result.error) {{
                        updateMeasureInfo('&#9888; ' + result.error);
                        return;
                    }}

                    if (result.scales && result.scales.length > 0) {{
                        // Found scales! Display them
                        const scaleList = result.scales.map(s => s.scale + ' (p.' + s.page + ')').join(', ');
                        updateMeasureInfo('&#10004; Found scales: ' + scaleList + '. Click to apply the first one.');

                        // Auto-apply the first scale found
                        const firstScale = result.scales[0].scale;

                        // Ask user to confirm
                        if (confirm('Found scale: ' + firstScale + '\\n\\nApply this scale to the current item?')) {{
                            // Save scale to item
                            fetch('/api/update-item', {{
                                method: 'POST',
                                headers: {{ 'Content-Type': 'application/json' }},
                                body: JSON.stringify({{
                                    id: item.id,
                                    fields: {{ scala_metrica: firstScale }}
                                }})
                            }})
                            .then(r => r.json())
                            .then(res => {{
                                if (res.success) {{
                                    item.scala_metrica = firstScale;
                                    updateMeasureInfo('&#10004; Scale ' + firstScale + ' saved! Use Calibrate for precise pixel calibration.');
                                    selectItem(currentIndex);  // Refresh display
                                }}
                            }});
                        }}
                    }} else {{
                        updateMeasureInfo('&#9888; No scale found in PDF. Use manual Calibrate instead.');
                    }}
                }})
                .catch(err => {{
                    updateMeasureInfo('&#9888; Error: ' + err);
                }});
        }}

        // ===== 3D VIEWER =====

        let viewer3dScene, viewer3dCamera, viewer3dRenderer, viewer3dControls;
        let viewer3dModel = null;
        let viewer3dEdges = null;
        let viewer3dGrid = null;
        let viewer3dLights = [];
        let viewer3dGlbData = null;
        let viewer3dGlbTexturedData = null;
        let viewer3dGlbAIData = null;
        let viewer3dAIAnalysis = null;
        let viewer3dShowTextured = true;
        let viewer3dShowAI = false;
        let viewer3dCurrentItem = null;

        function open3DViewer() {{
            const item = filteredData[currentIndex];
            if (!item) {{
                alert('No item selected');
                return;
            }}

            // Store current item and reset AI data
            viewer3dCurrentItem = item;
            viewer3dGlbAIData = null;
            viewer3dAIAnalysis = null;
            viewer3dShowAI = false;

            const modal = document.getElementById('viewer3dModal');
            const loading = document.getElementById('viewer3dLoading');
            const info = document.getElementById('viewer3dInfo');
            const aiInfo = document.getElementById('viewer3dAIInfo');

            modal.classList.add('active');
            loading.style.display = 'block';
            loading.innerHTML = '<div class="spinner"></div><p>Generating 3D model...</p>';
            info.textContent = 'Processing ' + item.id + '...';
            if (aiInfo) aiInfo.style.display = 'none';

            // Hide AI toggle button initially
            const toggleAIBtn = document.getElementById('toggleAIBtn');
            if (toggleAIBtn) toggleAIBtn.style.display = 'none';

            // Request 3D reconstruction
            fetch('/api/3d/reconstruct', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ image_path: item.image_path, with_decoration: true }})
            }})
            .then(r => r.json())
            .then(data => {{
                if (data.error) {{
                    loading.innerHTML = '<p style="color:#ff6b6b;">&#9888; ' + data.error + '</p>';
                    return;
                }}

                viewer3dGlbData = data.glb_data;
                viewer3dGlbTexturedData = data.glb_textured_data || null;

                let infoText = data.vertices.toLocaleString() + ' vertices, ' + data.faces.toLocaleString() + ' faces';
                if (data.has_thickness) {{
                    infoText += ' | Wall: ' + data.wall_thickness.toFixed(1) + 'px';
                }}
                if (data.has_decoration) {{
                    infoText += ' | Decoration detected';
                }}
                info.textContent = infoText;
                loading.style.display = 'none';

                // Show/hide decoration toggle button
                const toggleBtn = document.getElementById('toggleDecorationBtn');
                if (toggleBtn) {{
                    toggleBtn.style.display = viewer3dGlbTexturedData ? 'inline-block' : 'none';
                }}

                init3DViewer();
                load3DModel(viewer3dShowTextured && viewer3dGlbTexturedData ? viewer3dGlbTexturedData : viewer3dGlbData);
            }})
            .catch(err => {{
                loading.innerHTML = '<p style="color:#ff6b6b;">&#9888; ' + err + '</p>';
            }});
        }}

        function toggleDecoration() {{
            viewer3dShowTextured = !viewer3dShowTextured;
            viewer3dShowAI = false;
            const btn = document.getElementById('toggleDecorationBtn');
            if (btn) {{
                btn.textContent = viewer3dShowTextured ? '&#127912; Plain' : '&#127912; Decorated';
            }}
            if (viewer3dScene) {{
                load3DModel(viewer3dShowTextured && viewer3dGlbTexturedData ? viewer3dGlbTexturedData : viewer3dGlbData);
            }}
            // Hide AI info when switching to standard view
            const aiInfo = document.getElementById('viewer3dAIInfo');
            if (aiInfo) aiInfo.style.display = 'none';
        }}

        function toggleAIDecoration() {{
            viewer3dShowAI = !viewer3dShowAI;
            const btn = document.getElementById('toggleAIBtn');
            if (btn) {{
                btn.textContent = viewer3dShowAI ? '&#127912; Standard' : '&#129302; AI';
            }}
            if (viewer3dScene && viewer3dGlbAIData) {{
                load3DModel(viewer3dShowAI ? viewer3dGlbAIData : viewer3dGlbData);
            }}
            // Show/hide AI info panel
            const aiInfo = document.getElementById('viewer3dAIInfo');
            if (aiInfo) {{
                aiInfo.style.display = viewer3dShowAI ? 'block' : 'none';
            }}
        }}

        function openAIModal() {{
            // Check for saved API key
            const savedKey = localStorage.getItem('anthropic_api_key');
            if (savedKey) {{
                document.getElementById('anthropicApiKey').value = savedKey;
            }}
            document.getElementById('aiKeyModal').classList.add('active');
        }}

        function runAIAnalysis() {{
            const apiKey = document.getElementById('anthropicApiKey').value.trim();
            if (!apiKey) {{
                alert('Please enter your Anthropic API key');
                return;
            }}

            // Save key if checkbox is checked
            if (document.getElementById('saveApiKey').checked) {{
                localStorage.setItem('anthropic_api_key', apiKey);
            }} else {{
                localStorage.removeItem('anthropic_api_key');
            }}

            // Close modal
            closeModal('aiKeyModal');

            // Show loading
            const loading = document.getElementById('viewer3dLoading');
            const info = document.getElementById('viewer3dInfo');
            loading.style.display = 'block';
            loading.innerHTML = '<div class="spinner"></div><p>&#129302; Analyzing decoration with Claude AI...</p>';

            // Request AI analysis
            fetch('/api/3d/reconstruct', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    image_path: viewer3dCurrentItem.image_path,
                    with_decoration: true,
                    use_ai: true,
                    api_key: apiKey
                }})
            }})
            .then(r => r.json())
            .then(data => {{
                loading.style.display = 'none';
                console.log('AI Response:', data);

                if (data.error) {{
                    alert('AI Analysis error: ' + data.error);
                    return;
                }}

                // Always show AI analysis if available
                if (data.ai_analysis) {{
                    console.log('AI Analysis:', data.ai_analysis);
                    displayAIAnalysis(data.ai_analysis);
                    viewer3dAIAnalysis = data.ai_analysis;
                }}

                if (data.glb_ai_data) {{
                    viewer3dGlbAIData = data.glb_ai_data;

                    // Show toggle button
                    const toggleAIBtn = document.getElementById('toggleAIBtn');
                    if (toggleAIBtn) {{
                        toggleAIBtn.style.display = 'inline-block';
                    }}

                    // Switch to AI view with painting animation
                    viewer3dShowAI = true;
                    startPaintingAnimation(viewer3dGlbAIData);

                    info.textContent += ' | AI: ' + (data.ai_analysis?.decoration_type || 'analyzed');
                }} else if (data.ai_analysis) {{
                    // AI analysis succeeded but no texture - still show info
                    info.textContent += ' | AI analizzato (no texture)';
                    alert('Analisi AI completata. Vedi il pannello in alto a sinistra per i dettagli.');
                }} else {{
                    alert('Analisi AI non riuscita. Verifica la chiave API.');
                }}
            }})
            .catch(err => {{
                loading.style.display = 'none';
                alert('AI Analysis error: ' + err);
            }});
        }}

        // === PAINTING ANIMATION FOR RECONSTRUCTION ===
        let paintingAnimationId = null;

        function startPaintingAnimation(glbBase64) {{
            // Create painting overlay
            const container = document.getElementById('viewer3dContainer');

            // Create or get the painting overlay
            let overlay = document.getElementById('paintingOverlay');
            if (!overlay) {{
                overlay = document.createElement('div');
                overlay.id = 'paintingOverlay';
                overlay.innerHTML = `
                    <div class="painting-content">
                        <div class="brush-container">
                            <svg class="brush-icon" viewBox="0 0 64 64" width="48" height="48">
                                <path fill="#8B4513" d="M10,54 L16,48 L48,16 L54,22 L22,54 Z"/>
                                <path fill="#D2691E" d="M48,16 L54,10 L58,14 L54,22 Z"/>
                                <path fill="#FFD700" d="M54,10 L58,6 L62,10 L58,14 Z"/>
                                <circle fill="#333" cx="14" cy="52" r="3"/>
                            </svg>
                        </div>
                        <div class="painting-progress">
                            <div class="progress-bar"></div>
                        </div>
                        <div class="painting-text">Ricostruendo decorazione...</div>
                    </div>
                `;
                overlay.style.cssText = `
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0,0,0,0.7);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 100;
                    pointer-events: none;
                `;
                container.style.position = 'relative';
                container.appendChild(overlay);

                // Add CSS for animation
                const style = document.createElement('style');
                style.textContent = `
                    .painting-content {{
                        text-align: center;
                        color: white;
                    }}
                    .brush-container {{
                        animation: brush-move 2s ease-in-out infinite;
                    }}
                    @keyframes brush-move {{
                        0%, 100% {{ transform: translate(-30px, 0) rotate(-30deg); }}
                        25% {{ transform: translate(30px, -20px) rotate(-20deg); }}
                        50% {{ transform: translate(30px, 20px) rotate(-40deg); }}
                        75% {{ transform: translate(-30px, 10px) rotate(-35deg); }}
                    }}
                    .painting-progress {{
                        width: 200px;
                        height: 8px;
                        background: rgba(255,255,255,0.2);
                        border-radius: 4px;
                        margin: 20px auto;
                        overflow: hidden;
                    }}
                    .progress-bar {{
                        width: 0%;
                        height: 100%;
                        background: linear-gradient(90deg, #ffd700, #ff8c00);
                        border-radius: 4px;
                        transition: width 0.1s;
                    }}
                    .painting-text {{
                        font-size: 14px;
                        margin-top: 10px;
                    }}
                `;
                document.head.appendChild(style);
            }}

            overlay.style.display = 'flex';
            const progressBar = overlay.querySelector('.progress-bar');
            const paintingText = overlay.querySelector('.painting-text');

            // Animate progress
            let progress = 0;
            const messages = [
                'Ricostruendo decorazione...',
                'Applicando pattern...',
                'Specchiando motivi...',
                'Completando dettagli...',
                'Decorazione completata!'
            ];

            const animationInterval = setInterval(() => {{
                progress += 2;
                progressBar.style.width = progress + '%';

                // Update message
                const msgIndex = Math.min(Math.floor(progress / 25), messages.length - 1);
                paintingText.textContent = messages[msgIndex];

                if (progress >= 100) {{
                    clearInterval(animationInterval);

                    // Short delay then show the model
                    setTimeout(() => {{
                        overlay.style.display = 'none';
                        load3DModel(glbBase64);
                    }}, 500);
                }}
            }}, 50);  // 50ms * 50 steps = ~2.5 seconds total

            paintingAnimationId = animationInterval;
        }}

        function displayAIAnalysis(analysis) {{
            if (!analysis) return;

            const container = document.getElementById('aiAnalysisContent');
            const aiInfo = document.getElementById('viewer3dAIInfo');

            let html = '';

            // Period/Style (important for archaeology)
            if (analysis.period_style) {{
                html += '<div class="ai-field"><span class="ai-label">Stile/Periodo:</span> <span class="ai-value" style="color:#ffd700;">' + analysis.period_style + '</span></div>';
            }}

            // Basic info in a compact grid
            html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:5px;margin:8px 0;">';
            if (analysis.decoration_type) {{
                html += '<div class="ai-field"><span class="ai-label">Tipo:</span> <span class="ai-value">' + analysis.decoration_type + '</span></div>';
            }}
            if (analysis.technique) {{
                html += '<div class="ai-field"><span class="ai-label">Tecnica:</span> <span class="ai-value">' + analysis.technique + '</span></div>';
            }}
            if (analysis.symmetry) {{
                html += '<div class="ai-field"><span class="ai-label">Simmetria:</span> <span class="ai-value">' + analysis.symmetry + '</span></div>';
            }}
            if (analysis.repetition_module) {{
                html += '<div class="ai-field"><span class="ai-label">Modulo:</span> <span class="ai-value">' + analysis.repetition_module + '</span></div>';
            }}
            html += '</div>';

            // Tool marks
            if (analysis.tool_marks) {{
                html += '<div class="ai-field"><span class="ai-label">Strumento:</span> <span class="ai-value">' + analysis.tool_marks + '</span></div>';
            }}

            // Band structure
            if (analysis.band_structure && analysis.band_structure.length > 0) {{
                html += '<div class="ai-field" style="margin-top:8px;"><span class="ai-label">Struttura a fasce:</span>';
                html += '<div style="margin-top:5px;">';
                analysis.band_structure.forEach((band, idx) => {{
                    html += '<div style="background:rgba(255,255,255,0.1);padding:4px 8px;margin:3px 0;border-radius:4px;font-size:0.85em;">';
                    html += '<strong>' + (band.band_name || 'Fascia ' + (idx+1)) + '</strong>';
                    if (band.height_start !== undefined && band.height_end !== undefined) {{
                        html += ' <span style="opacity:0.7;">(' + band.height_start + '%-' + band.height_end + '%)</span>';
                    }}
                    if (band.main_motif) {{
                        html += '<br><span style="opacity:0.8;">' + band.main_motif + '</span>';
                    }}
                    html += '</div>';
                }});
                html += '</div></div>';
            }}

            // Patterns - detailed view
            if (analysis.patterns && analysis.patterns.length > 0) {{
                html += '<div class="ai-field" style="margin-top:8px;"><span class="ai-label">Pattern rilevati (' + analysis.patterns.length + '):</span>';
                html += '<div class="ai-patterns" style="margin-top:5px;">';
                analysis.patterns.forEach(p => {{
                    if (typeof p === 'string') {{
                        html += '<span class="ai-pattern-tag">' + p + '</span>';
                    }} else if (typeof p === 'object') {{
                        let patternText = p.type || 'pattern';
                        let tooltip = '';
                        if (p.position) {{
                            patternText += ' @' + p.position;
                        }}
                        if (p.height_percent_start !== undefined) {{
                            tooltip += p.height_percent_start + '%-' + p.height_percent_end + '% | ';
                        }}
                        if (p.line_count) {{
                            tooltip += p.line_count + ' linee | ';
                        }}
                        if (p.description_it) {{
                            tooltip += p.description_it;
                        }}
                        html += '<span class="ai-pattern-tag" title="' + tooltip + '" style="cursor:help;">' + patternText + '</span>';
                    }}
                }});
                html += '</div></div>';
            }}

            // Complete description
            const description = analysis.complete_description || analysis.description;
            if (description) {{
                html += '<div class="ai-field" style="margin-top:12px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.2);">';
                html += '<span class="ai-label">&#128221; Analisi Archeologica:</span>';
                html += '<div style="margin-top:8px;font-style:italic;opacity:0.95;line-height:1.5;font-size:0.9em;background:rgba(0,0,0,0.2);padding:10px;border-radius:6px;">' + description + '</div>';
                html += '</div>';
            }}

            container.innerHTML = html;
            aiInfo.style.display = 'block';

            // Make the panel scrollable and wider
            aiInfo.style.maxHeight = '450px';
            aiInfo.style.maxWidth = '380px';
            aiInfo.style.overflowY = 'auto';
        }}

        function init3DViewer() {{
            const container = document.getElementById('viewer3dContainer');

            // Clean up existing
            if (viewer3dRenderer) {{
                container.removeChild(viewer3dRenderer.domElement);
                viewer3dRenderer.dispose();
            }}

            // Scene
            viewer3dScene = new THREE.Scene();
            viewer3dScene.background = new THREE.Color(0x1a1a2e);

            // Camera
            const aspect = container.clientWidth / container.clientHeight;
            viewer3dCamera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
            viewer3dCamera.position.set(50, 30, 50);

            // Renderer
            viewer3dRenderer = new THREE.WebGLRenderer({{ antialias: true }});
            viewer3dRenderer.setSize(container.clientWidth, container.clientHeight);
            viewer3dRenderer.setPixelRatio(window.devicePixelRatio);
            container.appendChild(viewer3dRenderer.domElement);

            // Controls
            viewer3dControls = new THREE.OrbitControls(viewer3dCamera, viewer3dRenderer.domElement);
            viewer3dControls.enableDamping = true;
            viewer3dControls.dampingFactor = 0.05;

            // Lights - bright enough to see the model
            viewer3dLights = [];

            const ambientLight = new THREE.AmbientLight(0xffffff, 1.0);
            viewer3dScene.add(ambientLight);
            viewer3dLights.push(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
            directionalLight.position.set(50, 100, 50);
            viewer3dScene.add(directionalLight);
            viewer3dLights.push(directionalLight);

            const frontLight = new THREE.DirectionalLight(0xffffff, 0.8);
            frontLight.position.set(0, 50, 100);
            viewer3dScene.add(frontLight);
            viewer3dLights.push(frontLight);

            const backLight = new THREE.DirectionalLight(0xffffff, 0.5);
            backLight.position.set(-50, 50, -50);
            viewer3dScene.add(backLight);
            viewer3dLights.push(backLight);

            // Hemisphere light for softer shadows
            const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.6);
            viewer3dScene.add(hemiLight);
            viewer3dLights.push(hemiLight);

            // Grid
            viewer3dGrid = new THREE.GridHelper(100, 20, 0x4fc3f7, 0x333333);
            viewer3dScene.add(viewer3dGrid);

            console.log('3D Viewer initialized');

            // Animate
            function animate() {{
                if (!document.getElementById('viewer3dModal').classList.contains('active')) return;
                requestAnimationFrame(animate);
                viewer3dControls.update();
                viewer3dRenderer.render(viewer3dScene, viewer3dCamera);
            }}
            animate();

            // Handle resize
            window.addEventListener('resize', () => {{
                if (!viewer3dRenderer) return;
                const w = container.clientWidth;
                const h = container.clientHeight;
                viewer3dCamera.aspect = w / h;
                viewer3dCamera.updateProjectionMatrix();
                viewer3dRenderer.setSize(w, h);
            }});
        }}

        function load3DModel(glbBase64) {{
            const loader = new THREE.GLTFLoader();

            // Convert base64 to ArrayBuffer
            const binary = atob(glbBase64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) {{
                bytes[i] = binary.charCodeAt(i);
            }}

            loader.parse(bytes.buffer, '', function(gltf) {{
                console.log('GLB loaded successfully', gltf);

                // Remove old model and edges
                if (viewer3dModel) {{
                    viewer3dScene.remove(viewer3dModel);
                }}
                if (viewer3dEdges) {{
                    viewer3dScene.remove(viewer3dEdges);
                    viewer3dEdges = null;
                }}

                viewer3dModel = gltf.scene;

                // Get color from control panel
                const colorInput = document.getElementById('v3dColor');
                const modelColor = colorInput ? colorInput.value : '#ffffff';

                console.log('Applying color:', modelColor);

                // Store meshes for later control
                let meshCount = 0;

                // Apply material - preserve textures if present
                viewer3dModel.traverse(function(child) {{
                    console.log('Traversing:', child.type, child.name);
                    if (child.isMesh) {{
                        meshCount++;
                        console.log('Found mesh with geometry:', child.geometry);

                        // Check if original material has a texture
                        const hasTexture = child.material && child.material.map;
                        console.log('Has texture:', hasTexture);

                        if (hasTexture) {{
                            // Preserve texture, just update material properties
                            child.material.side = THREE.DoubleSide;
                            child.material.transparent = true;
                            child.material.opacity = 1.0;
                            // Make it brighter
                            if (child.material.color) {{
                                child.material.color.setHex(0xffffff);
                            }}
                            console.log('Preserved texture on material');
                        }} else {{
                            // No texture - use white ceramic material
                            const mat = new THREE.MeshStandardMaterial({{
                                color: modelColor,
                                side: THREE.DoubleSide,
                                transparent: true,
                                opacity: 1.0,
                                roughness: 0.7,
                                metalness: 0.1
                            }});
                            child.material = mat;
                        }}

                        // Store reference on mesh for later updates
                        child.userData.vesselMesh = true;
                        child.userData.hasTexture = hasTexture;

                        // Create edges
                        if (child.geometry) {{
                            const edgesGeometry = new THREE.EdgesGeometry(child.geometry, 15);
                            const edgeColorInput = document.getElementById('v3dEdgeColor');
                            const edgeColor = edgeColorInput ? edgeColorInput.value : '#000000';
                            const edgesMaterial = new THREE.LineBasicMaterial({{ color: edgeColor }});
                            const edges = new THREE.LineSegments(edgesGeometry, edgesMaterial);
                            edges.userData.isEdge = true;
                            child.add(edges);
                            console.log('Added edges to mesh');
                        }}
                    }}
                }});

                console.log('Total meshes found:', meshCount);

                // Center and scale
                const box = new THREE.Box3().setFromObject(viewer3dModel);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                console.log('Model size:', size);

                const maxDim = Math.max(size.x, size.y, size.z);
                const scale = maxDim > 0 ? 30 / maxDim : 1;

                viewer3dModel.scale.setScalar(scale);

                // Recalculate center after scaling
                const newBox = new THREE.Box3().setFromObject(viewer3dModel);
                const newCenter = newBox.getCenter(new THREE.Vector3());
                viewer3dModel.position.sub(newCenter);

                viewer3dScene.add(viewer3dModel);
                console.log('Model added to scene with edges');

                // Position camera
                viewer3dCamera.position.set(60, 40, 60);
                viewer3dControls.target.set(0, 0, 0);
                viewer3dControls.update();
            }},
            function(error) {{
                console.error('3D loading error:', error);
                document.getElementById('viewer3dLoading').innerHTML = '<p style="color:#ff6b6b;">Error loading 3D model: ' + error + '</p>';
                document.getElementById('viewer3dLoading').style.display = 'block';
            }});
        }}

        function reset3DView() {{
            if (viewer3dCamera && viewer3dControls) {{
                viewer3dCamera.position.set(50, 30, 50);
                viewer3dControls.target.set(0, 0, 0);
                viewer3dControls.update();
            }}
        }}

        function download3D() {{
            if (!viewer3dGlbData) return;

            const binary = atob(viewer3dGlbData);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) {{
                bytes[i] = binary.charCodeAt(i);
            }}

            const blob = new Blob([bytes], {{ type: 'model/gltf-binary' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const item = filteredData[currentIndex];
            a.download = (item ? item.id : 'vessel') + '.glb';
            a.click();
            URL.revokeObjectURL(url);
        }}

        function close3DViewer() {{
            document.getElementById('viewer3dModal').classList.remove('active');
            document.getElementById('viewer3dLoading').style.display = 'block';
            viewer3dGlbData = null;
        }}

        // Close 3D viewer on Escape
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape' && document.getElementById('viewer3dModal').classList.contains('active')) {{
                close3DViewer();
            }}
        }});

        // ===== 3D VIEWER CONTROLS =====

        function update3DColor(color) {{
            if (!viewer3dModel) return;
            console.log('Updating color to:', color);
            viewer3dModel.traverse(function(child) {{
                if (child.isMesh && child.material && child.userData.vesselMesh) {{
                    // For textured models, use emissive to tint the color
                    if (child.userData.hasTexture && child.material.emissive) {{
                        // Convert hex to RGB and apply as subtle tint
                        const c = new THREE.Color(color);
                        child.material.emissive = c;
                        child.material.emissiveIntensity = 0.15;
                        console.log('Applied emissive tint to textured mesh');
                    }} else {{
                        // For non-textured models, directly change color
                        child.material.color.setStyle(color);
                        console.log('Applied direct color to mesh');
                    }}
                    child.material.needsUpdate = true;
                }}
            }});
        }}

        function toggle3DWireframe(enabled) {{
            if (!viewer3dModel) return;
            console.log('Wireframe:', enabled);
            viewer3dModel.traverse(function(child) {{
                if (child.isMesh && child.material && child.userData.vesselMesh) {{
                    child.material.wireframe = enabled;
                    child.material.needsUpdate = true;
                }}
            }});
        }}

        function toggle3DEdges(show) {{
            if (!viewer3dModel) return;
            console.log('Edges:', show);
            viewer3dModel.traverse(function(child) {{
                if (child.userData.isEdge) {{
                    child.visible = show;
                }}
            }});
        }}

        function update3DEdgeColor(color) {{
            if (!viewer3dModel) return;
            console.log('Edge color:', color);
            let edgesUpdated = 0;
            viewer3dModel.traverse(function(child) {{
                // Check for edges - they are LineSegments with isEdge userData
                if (child.userData && child.userData.isEdge && child.material) {{
                    child.material.color.setStyle(color);
                    child.material.needsUpdate = true;
                    edgesUpdated++;
                }}
                // Also check for LineSegments type as backup
                if (child.isLineSegments && child.material) {{
                    child.material.color.setStyle(color);
                    child.material.needsUpdate = true;
                    edgesUpdated++;
                }}
            }});
            console.log('Edges updated:', edgesUpdated);
        }}

        function update3DOpacity(value) {{
            if (!viewer3dModel) return;
            const opacity = parseFloat(value);
            console.log('Opacity:', opacity);
            viewer3dModel.traverse(function(child) {{
                if (child.isMesh && child.material && child.userData.vesselMesh) {{
                    child.material.opacity = opacity;
                    child.material.transparent = true;
                    child.material.needsUpdate = true;
                }}
            }});
        }}

        function update3DLight(value) {{
            const intensity = parseFloat(value);
            viewer3dLights.forEach(light => {{
                if (light.isAmbientLight) {{
                    light.intensity = intensity;
                }} else if (light.isDirectionalLight) {{
                    light.intensity = intensity * 0.8;
                }} else if (light.isHemisphereLight) {{
                    light.intensity = intensity * 0.6;
                }}
            }});
        }}

        function toggle3DGrid(show) {{
            if (viewer3dGrid) {{
                viewer3dGrid.visible = show;
            }}
        }}

        function toggle3DAutoRotate(enabled) {{
            if (viewer3dControls) {{
                viewer3dControls.autoRotate = enabled;
                viewer3dControls.autoRotateSpeed = 2.0;
            }}
        }}

    </script>
</body>
</html>
'''


# ============================================================================
# MAIN
# ============================================================================

def main():
    os.chdir(Path(__file__).parent)

    # Initialize database
    init_db()
    run_auto_migrations()  # Auto-add new columns if missing
    sync_bundled_data()    # Sync new collections from bundled DB to persistent DB
    migrate_csv_to_db()

    # Load config
    config = load_config()

    db_location = DB_FILE if DATA_DIR else str(Path(__file__).parent / DB_FILE)
    persistence = "PERSISTENT (Railway Volume)" if DATA_DIR else "Local"

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║             CeramicaDatabase - Unified Viewer v2.0                   ║
╠══════════════════════════════════════════════════════════════════════╣
║  Server: http://{HOST}:{PORT}                                            ║
║  Database: {persistence}                                             ║
║  DB Path: {db_location[:50]}                                         ║
║                                                                      ║
║  Credentials:                                                        ║
║    Admin:  admin2024   (full access)                                 ║
║    Viewer: viewer2024  (read-only)                                   ║
║                                                                      ║
║  API Endpoints (public):                                             ║
║    GET /api/v1/items      - All items                                ║
║    GET /api/v1/items/{{id}} - Single item                              ║
║    GET /api/v1/vocabulary - Controlled vocabulary                    ║
║    GET /api/v1/periods    - All periods                              ║
║    GET /api/v1/stats      - Statistics                               ║
║                                                                      ║
║  Press Ctrl+C to stop                                                ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Open browser if local
    if not IS_RAILWAY:
        webbrowser.open(f'http://localhost:{PORT}')

    server = HTTPServer((HOST, PORT), ViewerHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.shutdown()


if __name__ == '__main__':
    main()
