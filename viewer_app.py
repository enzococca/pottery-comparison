#!/usr/bin/env python3
"""
CeramicaDatabase - Viewer Unificato Multi-Collezione
Visualizzatore per ceramica archeologica con supporto multi-collezione

Esegui con: python viewer_app.py
"""

import os
import sys
import json
import subprocess
import webbrowser
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

# Configuration
PORT = int(os.environ.get('PORT', 8080))
HOST = os.environ.get('HOST', '0.0.0.0')
IS_RAILWAY = os.environ.get('RAILWAY_ENVIRONMENT') is not None

# Authentication - Password hash (SHA256 of 'ceramica2024')
PASSWORD_HASH = os.environ.get('PASSWORD_HASH', '9a617485652c16ed32e08d85cf3ea24e935cbe8afe9c650266392163a485b518')
# Session storage (in-memory, will reset on restart)
SESSIONS = {}
CONFIG_FILE = "config.json"
METADATA_FILE = "ceramica_metadata.csv"
EXCEL_FILE = "ceramica_metadata.xlsx"

# Macro-period mappings
MACRO_PERIODS = {
    "Umm an-Nar": [
        "umm an-nar", "umm-an-nar", "hili", "2700-2000", "2500-2000",
        "iii millennio", "iiie millénaire", "prima eta del bronzo",
        "early bronze", "hili ii"
    ],
    "Wadi Suq": [
        "wadi suq", "wadi-suq", "2000-1600", "2000-1800", "1800-1600",
        "bronze moyen", "middle bronze"
    ],
    "Late Bronze Age": [
        "bronze récent", "bronze recent", "late bronze", "1600-1250",
        "1600-600", "fer i", "fer ii", "iron age", "fer ", "masafi"
    ]
}

# Default configuration
DEFAULT_CONFIG = {
    "collections": {
        "Degli_Espositi": {
            "name": "Degli Espositi - Tesi MDE",
            "pdf": "PDFs/2- Capp.4-5-6+bibliografia.pdf",
            "color": "#4472C4"
        },
        "Righetti": {
            "name": "Righetti - Hili 8 / Wadi Suq",
            "pdf": "PDFs/Righetti_Thèse_Volume_II.pdf",
            "color": "#ED7D31"
        },
        "Pellegrino": {
            "name": "Pellegrino - Masafi / Dibba / Tell Abraq",
            "pdf": "PDFs/2021-11_Pellegrino_cér.pdf",
            "color": "#70AD47"
        }
    },
    "metadata_file": "ceramica_metadata.csv"
}


def get_macro_period(period_str):
    """Determine macro-period from period string"""
    if not period_str:
        return ""
    period_lower = period_str.lower()
    for macro, keywords in MACRO_PERIODS.items():
        for keyword in keywords:
            if keyword in period_lower:
                return macro
    return ""


def rotate_image(image_path, degrees):
    """Rotate image by specified degrees and save"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False

        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, degrees, 1.0)

        # Calculate new bounding box
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust rotation matrix
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Rotate with white background
        rotated = cv2.warpAffine(img, M, (new_w, new_h),
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(255, 255, 255))

        cv2.imwrite(image_path, rotated)
        return True
    except Exception as e:
        print(f"Rotation error: {e}")
        return False


def verify_password(password):
    """Verify password against stored hash"""
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return password_hash == PASSWORD_HASH


def create_session():
    """Create a new session token"""
    token = secrets.token_hex(32)
    SESSIONS[token] = True
    return token


def verify_session(cookie_header):
    """Verify session from cookie header"""
    if not cookie_header:
        return False
    cookies = http.cookies.SimpleCookie()
    try:
        cookies.load(cookie_header)
        if 'session' in cookies:
            return cookies['session'].value in SESSIONS
    except:
        pass
    return False


# Login page HTML
LOGIN_PAGE = '''<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CeramicaDatabase - Login</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #e0e0e0;
        }
        .login-container {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 50px;
            box-shadow: 0 25px 45px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
            max-width: 450px;
            width: 90%;
        }
        .logo {
            font-size: 4em;
            margin-bottom: 20px;
        }
        h1 {
            color: #4fc3f7;
            font-size: 2em;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #888;
            margin-bottom: 30px;
            font-size: 0.95em;
        }
        .stats-box {
            background: rgba(79, 195, 247, 0.1);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }
        .stat {
            text-align: center;
        }
        .stat-number {
            font-size: 1.8em;
            font-weight: bold;
            color: #4fc3f7;
        }
        .stat-label {
            font-size: 0.75em;
            color: #888;
            margin-top: 5px;
        }
        .collections {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        .collection-badge {
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 500;
        }
        .login-form {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        input[type="password"] {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #fff;
            padding: 15px 20px;
            border-radius: 10px;
            font-size: 1em;
            text-align: center;
            transition: all 0.3s ease;
        }
        input[type="password"]:focus {
            outline: none;
            border-color: #4fc3f7;
            box-shadow: 0 0 15px rgba(79, 195, 247, 0.3);
        }
        input[type="password"]::placeholder {
            color: #666;
        }
        button {
            background: linear-gradient(135deg, #4fc3f7 0%, #29b6f6 100%);
            color: #000;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(79, 195, 247, 0.3);
        }
        .error {
            color: #ff5252;
            font-size: 0.9em;
            display: none;
        }
        .error.show {
            display: block;
        }
        .footer {
            margin-top: 30px;
            font-size: 0.8em;
            color: #666;
        }
        .footer a {
            color: #4fc3f7;
            text-decoration: none;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="logo">🏺</div>
        <h1>CeramicaDatabase</h1>
        <p class="subtitle">Sistema Unificato per Collezioni di Ceramica Archeologica</p>

        <div class="stats-box">
            <div class="stat">
                <div class="stat-number">797</div>
                <div class="stat-label">Elementi</div>
            </div>
            <div class="stat">
                <div class="stat-number">3</div>
                <div class="stat-label">Collezioni</div>
            </div>
            <div class="stat">
                <div class="stat-number">3</div>
                <div class="stat-label">Periodi</div>
            </div>
        </div>

        <div class="collections">
            <span class="collection-badge" style="background: rgba(68, 114, 196, 0.3); color: #4472C4;">Degli Espositi</span>
            <span class="collection-badge" style="background: rgba(237, 125, 49, 0.3); color: #ED7D31;">Righetti</span>
            <span class="collection-badge" style="background: rgba(112, 173, 71, 0.3); color: #70AD47;">Pellegrino</span>
        </div>

        <form class="login-form" onsubmit="login(event)">
            <input type="password" id="password" placeholder="Inserisci password" autocomplete="current-password">
            <p class="error" id="error">Password non corretta</p>
            <button type="submit">Accedi al Database</button>
        </form>

        <div class="footer">
            <p>Sviluppato da <a href="https://github.com/enzococca" target="_blank">Enzo Cocca</a></p>
        </div>
    </div>

    <script>
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
                error.textContent = 'Errore di connessione';
                error.classList.add('show');
            }
        }
    </script>
</body>
</html>
'''


class ViewerHandler(SimpleHTTPRequestHandler):
    def check_auth(self):
        """Check if request is authenticated"""
        cookie_header = self.headers.get('Cookie')
        return verify_session(cookie_header)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        # Root path - show login page
        if parsed.path == '/' or parsed.path == '/login':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(LOGIN_PAGE.encode('utf-8'))
            return

        # Viewer path - requires authentication
        if parsed.path == '/viewer':
            if not self.check_auth():
                self.send_response(302)
                self.send_header('Location', '/')
                self.end_headers()
                return

            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode('utf-8'))
            return

        # API endpoint to get configuration
        if parsed.path == '/api/config':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            config = load_config()
            config['macro_periods'] = list(MACRO_PERIODS.keys())
            self.wfile.write(json.dumps(config, ensure_ascii=False).encode('utf-8'))
            return

        # API endpoint to get data
        if parsed.path == '/api/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            try:
                config = load_config()
                df = pd.read_csv(config.get('metadata_file', METADATA_FILE))
                df = df.fillna('')
                # Add macro_period column
                df['macro_period'] = df['period'].apply(get_macro_period)
                data = df.to_dict('records')
                self.wfile.write(json.dumps(data, default=str, ensure_ascii=False).encode('utf-8'))
            except Exception as e:
                self.wfile.write(json.dumps({'error': str(e)}).encode())
            return

        # API endpoint to open PDF at specific page
        if parsed.path.startswith('/api/open-pdf'):
            query = urllib.parse.parse_qs(parsed.query)
            page = query.get('page', ['1'])[0]
            collection = query.get('collection', [''])[0]

            # Extract page number
            page_match = re.search(r'p+\.\s*(\d+)', page)
            if page_match:
                page_num = page_match.group(1)
            else:
                page_num = '1'

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            try:
                if sys.platform == 'darwin':
                    config = load_config()

                    # Get PDF path based on collection
                    pdf_path = None
                    if collection and collection in config.get('collections', {}):
                        pdf_path = config['collections'][collection].get('pdf', '')

                    if pdf_path:
                        base_path = Path(__file__).parent
                        full_pdf_path = str(base_path / pdf_path)

                        script = f'''
                        do shell script "open -b com.apple.Preview " & quoted form of "{full_pdf_path}"

                        delay 2.5

                        tell application "Preview"
                            activate
                            delay 1
                        end tell

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
                        self.wfile.write(json.dumps({
                            'success': True,
                            'page': page_num,
                            'pdf': full_pdf_path,
                            'collection': collection
                        }).encode())
                    else:
                        self.wfile.write(json.dumps({'error': 'PDF not found for collection'}).encode())
                else:
                    self.wfile.write(json.dumps({'error': 'Not supported on this platform'}).encode())
            except Exception as e:
                self.wfile.write(json.dumps({'error': str(e)}).encode())
            return

        return SimpleHTTPRequestHandler.do_GET(self)

    def do_DELETE(self):
        """Handle DELETE requests for image deletion"""
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == '/api/delete-image':
            query = urllib.parse.parse_qs(parsed.query)
            image_path = query.get('path', [''])[0]
            item_id = query.get('id', [''])[0]

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            try:
                base_path = Path(__file__).parent
                result = delete_image(base_path, image_path, item_id)
                self.wfile.write(json.dumps(result, ensure_ascii=False).encode('utf-8'))
            except Exception as e:
                self.wfile.write(json.dumps({'error': str(e)}).encode())
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        """Handle POST requests"""
        parsed = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b'{}'

        # Login endpoint
        if parsed.path == '/api/login':
            login_data = json.loads(post_data.decode('utf-8'))
            password = login_data.get('password', '')

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')

            if verify_password(password):
                session_token = create_session()
                self.send_header('Set-Cookie', f'session={session_token}; Path=/; HttpOnly; SameSite=Strict')
                self.end_headers()
                self.wfile.write(json.dumps({'success': True}).encode())
            else:
                self.end_headers()
                self.wfile.write(json.dumps({'success': False, 'error': 'Invalid password'}).encode())
            return

        # Batch delete
        if parsed.path == '/api/delete-batch':
            items = json.loads(post_data.decode('utf-8'))

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            try:
                base_path = Path(__file__).parent
                results = {'deleted': 0, 'errors': []}

                for item in items:
                    try:
                        result = delete_image(base_path, item.get('path', ''), item.get('id', ''))
                        if result['success']:
                            results['deleted'] += 1
                        else:
                            results['errors'].append(item.get('id', 'unknown'))
                    except Exception as e:
                        results['errors'].append(f"{item.get('id', 'unknown')}: {str(e)}")

                results['success'] = results['deleted'] > 0
                self.wfile.write(json.dumps(results, ensure_ascii=False).encode('utf-8'))
            except Exception as e:
                self.wfile.write(json.dumps({'error': str(e)}).encode())
            return

        # Update single item
        if parsed.path == '/api/update-item':
            update_data = json.loads(post_data.decode('utf-8'))

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            try:
                result = update_item(update_data)
                self.wfile.write(json.dumps(result, ensure_ascii=False).encode('utf-8'))
            except Exception as e:
                self.wfile.write(json.dumps({'error': str(e)}).encode())
            return

        # Batch update
        if parsed.path == '/api/update-batch':
            batch_data = json.loads(post_data.decode('utf-8'))

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            try:
                result = update_batch(batch_data)
                self.wfile.write(json.dumps(result, ensure_ascii=False).encode('utf-8'))
            except Exception as e:
                self.wfile.write(json.dumps({'error': str(e)}).encode())
            return

        # Rotate image
        if parsed.path == '/api/rotate-image':
            rotate_data = json.loads(post_data.decode('utf-8'))

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            try:
                base_path = Path(__file__).parent
                image_path = str(base_path / rotate_data.get('path', ''))
                degrees = rotate_data.get('degrees', 90)

                success = rotate_image(image_path, degrees)
                self.wfile.write(json.dumps({'success': success}).encode('utf-8'))
            except Exception as e:
                self.wfile.write(json.dumps({'error': str(e)}).encode())
            return

        self.send_response(404)
        self.end_headers()

    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()


def update_item(update_data):
    """Update a single item in the database"""
    item_id = update_data.get('id')
    fields = update_data.get('fields', {})

    if not item_id or not fields:
        return {'success': False, 'error': 'Missing id or fields'}

    base_path = Path(__file__).parent
    csv_path = base_path / METADATA_FILE

    df = pd.read_csv(csv_path)
    idx = df[df['id'] == item_id].index

    if len(idx) == 0:
        return {'success': False, 'error': 'Item not found'}

    for field, value in fields.items():
        if field in df.columns:
            df.loc[idx, field] = value

    df.to_csv(csv_path, index=False)

    # Update Excel
    try:
        df.to_excel(base_path / EXCEL_FILE, index=False)
    except:
        pass

    return {'success': True, 'updated': item_id}


def update_batch(batch_data):
    """Update multiple items with the same values"""
    item_ids = batch_data.get('ids', [])
    fields = batch_data.get('fields', {})

    if not item_ids or not fields:
        return {'success': False, 'error': 'Missing ids or fields'}

    base_path = Path(__file__).parent
    csv_path = base_path / METADATA_FILE

    df = pd.read_csv(csv_path)
    updated = 0

    for item_id in item_ids:
        idx = df[df['id'] == item_id].index
        if len(idx) > 0:
            for field, value in fields.items():
                if field in df.columns:
                    df.loc[idx, field] = value
            updated += 1

    df.to_csv(csv_path, index=False)

    # Update Excel
    try:
        df.to_excel(base_path / EXCEL_FILE, index=False)
    except:
        pass

    return {'success': True, 'updated': updated}


def delete_image(base_path, image_path, item_id):
    """Delete image file and remove record from CSV/Excel"""
    results = {'success': False, 'deleted_file': False, 'updated_csv': False, 'updated_excel': False}

    # Delete the image file
    full_image_path = base_path / image_path
    if full_image_path.exists():
        os.remove(full_image_path)
        results['deleted_file'] = True
        print(f"Deleted file: {full_image_path}")

    # Update CSV
    csv_path = base_path / METADATA_FILE
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        original_len = len(df)
        df = df[df['id'] != item_id]
        if len(df) < original_len:
            df.to_csv(csv_path, index=False)
            results['updated_csv'] = True
            print(f"Updated CSV: removed {item_id}")

    # Update Excel
    excel_path = base_path / EXCEL_FILE
    if excel_path.exists():
        try:
            df = pd.read_csv(csv_path)
            df.to_excel(excel_path, index=False, sheet_name='Metadati')
            results['updated_excel'] = True
        except Exception as e:
            print(f"Excel update error: {e}")

    results['success'] = results['deleted_file'] or results['updated_csv']
    return results


def load_config():
    """Load configuration from file or return defaults"""
    config_path = Path(__file__).parent / CONFIG_FILE
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return DEFAULT_CONFIG


def save_config(config):
    """Save configuration to file"""
    config_path = Path(__file__).parent / CONFIG_FILE
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


HTML_CONTENT = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CeramicaDatabase - Unified Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }
        .header {
            background: rgba(0, 0, 0, 0.4);
            padding: 12px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .header h1 {
            color: #4fc3f7;
            font-size: 1.4em;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .header-buttons {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .collection-tabs {
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
        }
        .collection-tab {
            padding: 8px 16px;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.85em;
            transition: all 0.2s ease;
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
        }
        .collection-tab.active {
            background: var(--tab-color, #4fc3f7);
            color: #000;
            font-weight: bold;
        }
        .collection-tab:hover:not(.active) {
            background: rgba(255, 255, 255, 0.2);
        }
        .controls {
            background: rgba(0, 0, 0, 0.2);
            padding: 10px 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .control-group {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .controls label { color: #888; font-size: 0.75em; }
        .controls select, .controls input {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #fff;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 0.8em;
        }
        .stats {
            margin-left: auto;
            background: rgba(79, 195, 247, 0.15);
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.85em;
            color: #4fc3f7;
        }
        .main-container { display: flex; height: calc(100vh - 130px); }
        .sidebar {
            width: 280px;
            background: rgba(0, 0, 0, 0.25);
            overflow-y: auto;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        .item-list { padding: 8px; }
        .item-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 8px;
            margin-bottom: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
            border-left: 3px solid transparent;
            display: flex;
            gap: 8px;
        }
        .item-card:hover { background: rgba(255, 255, 255, 0.1); }
        .item-card.active {
            background: rgba(79, 195, 247, 0.15);
            border-left-color: var(--collection-color, #4fc3f7);
        }
        .item-card img {
            width: 50px;
            height: 50px;
            object-fit: cover;
            border-radius: 4px;
            flex-shrink: 0;
        }
        .item-card .info { flex: 1; min-width: 0; overflow: hidden; }
        .item-card .id {
            font-weight: 600;
            color: #fff;
            font-size: 0.8em;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .item-card .period {
            font-size: 0.7em;
            color: #ff9800;
            margin-top: 2px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .item-card .meta {
            font-size: 0.65em;
            color: #888;
            margin-top: 2px;
        }
        .item-card .tags {
            display: flex;
            gap: 3px;
            margin-top: 3px;
            flex-wrap: wrap;
        }
        .tag {
            font-size: 0.55em;
            padding: 1px 5px;
            border-radius: 8px;
            background: rgba(255,255,255,0.1);
        }
        .tag.decorated { background: #e91e63; color: white; }
        .tag.plain { background: #607d8b; color: white; }
        .tag.vessel { background: #2196f3; color: white; }
        .content { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
        .image-viewer {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 15px;
            background: rgba(0, 0, 0, 0.3);
            position: relative;
        }
        .image-viewer img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
        }
        .nav-btn {
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
        }
        .nav-btn:hover { background: rgba(79, 195, 247, 0.4); }
        .nav-btn.prev { left: 10px; }
        .nav-btn.next { right: 10px; }
        .rotate-btns {
            position: absolute;
            top: 10px;
            right: 10px;
            display: flex;
            gap: 5px;
        }
        .rotate-btn {
            background: rgba(156, 39, 176, 0.6);
            border: none;
            color: white;
            padding: 8px 12px;
            cursor: pointer;
            border-radius: 5px;
            font-size: 1em;
        }
        .rotate-btn:hover { background: rgba(156, 39, 176, 0.9); }
        .metadata-panel {
            background: rgba(0, 0, 0, 0.4);
            padding: 12px 15px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            max-height: 220px;
            overflow-y: auto;
        }
        .metadata-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 8px;
        }
        .meta-item {
            background: rgba(255, 255, 255, 0.05);
            padding: 8px 10px;
            border-radius: 5px;
            border-left: 3px solid #4fc3f7;
        }
        .meta-item .label {
            font-size: 0.65em;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 2px;
        }
        .meta-item .value {
            font-size: 0.85em;
            color: #fff;
            word-break: break-word;
        }
        .meta-item.editable .value {
            cursor: pointer;
            border-bottom: 1px dashed rgba(255,255,255,0.3);
        }
        .meta-item.editable .value:hover {
            background: rgba(255,255,255,0.1);
        }
        .meta-item.period { border-left-color: #ff9800; }
        .meta-item.period .value { color: #ff9800; font-weight: bold; }
        .meta-item.decoration { border-left-color: #e91e63; }
        .meta-item.decoration .value { color: #e91e63; }
        .meta-item.vessel_type { border-left-color: #2196f3; }
        .meta-item.vessel_type .value { color: #2196f3; }
        .meta-item.part_type { border-left-color: #9c27b0; }
        .meta-item.part_type .value { color: #9c27b0; }
        .meta-item.page-ref { border-left-color: #4caf50; }
        .meta-item.page-ref .value {
            color: #4caf50;
            cursor: pointer;
            text-decoration: underline;
        }
        .meta-item.collection { border-left-color: var(--collection-color, #4fc3f7); }
        .action-btn {
            border: none;
            color: white;
            padding: 6px 12px;
            border-radius: 18px;
            cursor: pointer;
            font-size: 0.75em;
            transition: all 0.2s ease;
        }
        .action-btn:hover { transform: scale(1.05); }
        .pdf-btn { background: linear-gradient(135deg, #4caf50, #45a049); }
        .delete-btn { background: linear-gradient(135deg, #f44336, #d32f2f); }
        .select-btn { background: linear-gradient(135deg, #9c27b0, #7b1fa2); }
        .select-btn.active { background: linear-gradient(135deg, #ff9800, #f57c00); }
        .edit-btn { background: linear-gradient(135deg, #2196f3, #1976d2); }
        .batch-btn {
            background: linear-gradient(135deg, #ff5722, #e64a19);
            display: none;
        }
        .batch-btn.visible { display: inline-block; }
        .no-image { color: #666; text-align: center; padding: 20px; }
        .item-checkbox {
            width: 18px;
            height: 18px;
            cursor: pointer;
            accent-color: #9c27b0;
            flex-shrink: 0;
            display: none;
        }
        .select-mode .item-checkbox { display: block; }
        .item-card.selected {
            background: rgba(156, 39, 176, 0.25);
            border-left-color: #9c27b0;
        }
        .selection-count {
            background: rgba(156, 39, 176, 0.3);
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.75em;
            color: #ce93d8;
            display: none;
        }
        .selection-count.visible { display: inline-block; }
        .loading {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 30px;
            color: #4fc3f7;
        }
        .loading-spinner {
            width: 35px;
            height: 35px;
            border: 3px solid rgba(79, 195, 247, 0.2);
            border-top-color: #4fc3f7;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 12px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Modal styles */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        .modal-overlay.active { display: flex; }
        .modal {
            background: #2a2a4a;
            padding: 25px;
            border-radius: 12px;
            max-width: 500px;
            width: 90%;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        }
        .modal h3 { color: #4fc3f7; margin-bottom: 15px; }
        .modal p { margin-bottom: 15px; color: #ccc; }
        .modal-buttons { display: flex; gap: 10px; justify-content: center; margin-top: 20px; }
        .modal-btn {
            padding: 10px 25px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.2s;
        }
        .modal-btn.cancel { background: #555; color: #fff; }
        .modal-btn.confirm { background: #4caf50; color: #fff; }
        .modal-btn.danger { background: #f44336; color: #fff; }
        .modal-btn:hover { opacity: 0.85; }
        .edit-form { display: flex; flex-direction: column; gap: 12px; }
        .edit-form label { color: #888; font-size: 0.8em; }
        .edit-form select, .edit-form input {
            width: 100%;
            padding: 8px;
            border-radius: 5px;
            border: 1px solid rgba(255,255,255,0.2);
            background: rgba(255,255,255,0.1);
            color: white;
        }
        .edit-row {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .edit-row label { width: 100px; flex-shrink: 0; }
        .edit-row select, .edit-row input { flex: 1; }
    </style>
</head>
<body>
    <div class="header">
        <h1>&#127994; CeramicaDatabase</h1>
        <div class="collection-tabs" id="collectionTabs"></div>
        <div class="header-buttons">
            <span class="selection-count" id="selectionCount">0 sel.</span>
            <button class="action-btn pdf-btn" onclick="openPdfAtPage()">&#128196; PDF</button>
            <button class="action-btn select-btn" id="selectBtn" onclick="toggleSelectMode()">&#9745; Select</button>
            <button class="action-btn batch-btn" id="batchEditBtn" onclick="openBatchEditModal()">&#9998; Edit Sel.</button>
            <button class="action-btn batch-btn" id="batchDeleteBtn" onclick="confirmBatchDelete()">&#128465; Delete Sel.</button>
            <button class="action-btn edit-btn" onclick="openEditModal()">&#9998; Edit</button>
            <button class="action-btn delete-btn" onclick="confirmDelete()">&#128465;</button>
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
            <label>Decoration:</label>
            <select id="filterDecoration">
                <option value="">All</option>
                <option value="decorated">Decorated</option>
                <option value="plain">Plain</option>
            </select>
        </div>
        <div class="control-group">
            <label>Vessel Type:</label>
            <select id="filterVesselType">
                <option value="">All</option>
                <option value="jar">Jar</option>
                <option value="bowl">Bowl</option>
                <option value="cup">Cup</option>
                <option value="plate">Plate</option>
                <option value="pot">Pot</option>
            </select>
        </div>
        <div class="control-group">
            <label>Part:</label>
            <select id="filterPartType">
                <option value="">All</option>
                <option value="rim">Rim</option>
                <option value="base">Base</option>
                <option value="wall">Wall</option>
                <option value="complete">Complete</option>
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
                <div class="rotate-btns">
                    <button class="rotate-btn" onclick="rotateImage(-90)" title="Rotate left">&#8634;</button>
                    <button class="rotate-btn" onclick="rotateImage(90)" title="Rotate right">&#8635;</button>
                    <button class="rotate-btn" onclick="rotateImage(180)" title="Flip">&#8693;</button>
                </div>
            </div>
            <div class="metadata-panel">
                <div class="metadata-grid" id="metadataGrid"></div>
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
                    <select id="editDecoration">
                        <option value="plain">Plain</option>
                        <option value="decorated">Decorated</option>
                    </select>
                </div>
                <div class="edit-row">
                    <label>Vessel Type:</label>
                    <select id="editVesselType">
                        <option value="jar">Jar</option>
                        <option value="bowl">Bowl</option>
                        <option value="cup">Cup</option>
                        <option value="plate">Plate</option>
                        <option value="pot">Pot</option>
                        <option value="unknown">Unknown</option>
                    </select>
                </div>
                <div class="edit-row">
                    <label>Part:</label>
                    <select id="editPartType">
                        <option value="rim">Rim</option>
                        <option value="base">Base</option>
                        <option value="wall">Wall</option>
                        <option value="complete">Complete</option>
                        <option value="fragment">Fragment</option>
                    </select>
                </div>
                <div class="edit-row">
                    <label>Period:</label>
                    <input type="text" id="editPeriod" placeholder="Chronological period">
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
                    <select id="batchDecoration">
                        <option value="">-- Don't change --</option>
                        <option value="plain">Plain</option>
                        <option value="decorated">Decorated</option>
                    </select>
                </div>
                <div class="edit-row">
                    <label>Vessel Type:</label>
                    <select id="batchVesselType">
                        <option value="">-- Don't change --</option>
                        <option value="jar">Jar</option>
                        <option value="bowl">Bowl</option>
                        <option value="cup">Cup</option>
                        <option value="plate">Plate</option>
                        <option value="pot">Pot</option>
                    </select>
                </div>
                <div class="edit-row">
                    <label>Part:</label>
                    <select id="batchPartType">
                        <option value="">-- Don't change --</option>
                        <option value="rim">Rim</option>
                        <option value="base">Base</option>
                        <option value="wall">Wall</option>
                        <option value="complete">Complete</option>
                    </select>
                </div>
            </div>
            <div class="modal-buttons">
                <button class="modal-btn cancel" onclick="closeModal('batchEditModal')">Cancel</button>
                <button class="modal-btn confirm" onclick="saveBatchEdit()">Save All</button>
            </div>
        </div>
    </div>

    <script>
        let data = [];
        let filteredData = [];
        let currentIndex = 0;
        let config = {};
        let activeCollection = 'all';
        let selectMode = false;
        let selectedItems = new Set();

        // Load config and data
        Promise.all([
            fetch('/api/config').then(r => r.json()),
            fetch('/api/data').then(r => r.json())
        ]).then(([cfg, d]) => {
            config = cfg;
            data = d;
            initializeViewer();
        }).catch(err => {
            document.getElementById('itemList').innerHTML =
                '<div class="loading"><p>Errore: ' + err + '</p></div>';
        });

        function initializeViewer() {
            const tabs = document.getElementById('collectionTabs');
            tabs.innerHTML = '<button class="collection-tab active" data-collection="all">Tutti</button>';

            for (const [key, col] of Object.entries(config.collections || {})) {
                tabs.innerHTML += `<button class="collection-tab" data-collection="${key}" style="--tab-color: ${col.color}">${col.name}</button>`;
            }

            tabs.querySelectorAll('.collection-tab').forEach(tab => {
                tab.addEventListener('click', () => {
                    tabs.querySelectorAll('.collection-tab').forEach(t => t.classList.remove('active'));
                    tab.classList.add('active');
                    activeCollection = tab.dataset.collection;
                    applyFilters();
                });
            });

            // Filter listeners
            ['filterMacroPeriod', 'filterDecoration', 'filterVesselType', 'filterPartType'].forEach(id => {
                document.getElementById(id).addEventListener('change', applyFilters);
            });
            document.getElementById('searchInput').addEventListener('input', applyFilters);

            applyFilters();
        }

        function getCollectionColor(collection) {
            return config.collections?.[collection]?.color || '#4fc3f7';
        }

        function applyFilters() {
            const macroPeriod = document.getElementById('filterMacroPeriod').value;
            const decoration = document.getElementById('filterDecoration').value;
            const vesselType = document.getElementById('filterVesselType').value;
            const partType = document.getElementById('filterPartType').value;
            const search = document.getElementById('searchInput').value.toLowerCase();

            filteredData = data.filter(item => {
                if (activeCollection !== 'all' && item.collection !== activeCollection) return false;
                if (macroPeriod && item.macro_period !== macroPeriod) return false;
                if (decoration && item.decoration !== decoration) return false;
                if (vesselType && item.vessel_type !== vesselType) return false;
                if (partType && item.part_type !== partType) return false;
                if (search && !JSON.stringify(item).toLowerCase().includes(search)) return false;
                return true;
            });

            document.getElementById('stats').textContent = `${filteredData.length} / ${data.length}`;
            renderList();
            if (filteredData.length > 0) selectItem(0);
            else {
                document.getElementById('imageViewer').innerHTML = '<div class="no-image"><p>No items found</p></div><div class="rotate-btns"></div>';
                document.getElementById('metadataGrid').innerHTML = '';
            }
        }

        function renderList() {
            const list = document.getElementById('itemList');
            if (filteredData.length === 0) {
                list.innerHTML = '<div class="no-image"><p>No items</p></div>';
                return;
            }

            list.innerHTML = filteredData.map((item, i) => `
                <div class="item-card ${i === currentIndex ? 'active' : ''} ${selectedItems.has(item.id) ? 'selected' : ''}"
                     onclick="handleCardClick(event, ${i})"
                     style="--collection-color: ${getCollectionColor(item.collection)}">
                    <input type="checkbox" class="item-checkbox"
                           ${selectedItems.has(item.id) ? 'checked' : ''}
                           onclick="toggleItemSelection(event, '${item.id}')">
                    <img src="${item.image_path || ''}" onerror="this.style.display='none'">
                    <div class="info">
                        <div class="id">${item.id || 'N/A'}</div>
                        <div class="period">${(item.macro_period || item.period || '').substring(0, 30)}</div>
                        <div class="tags">
                            <span class="tag ${item.decoration || ''}">${item.decoration || '?'}</span>
                            <span class="tag vessel">${item.vessel_type || '?'}</span>
                        </div>
                    </div>
                </div>
            `).join('');

            if (selectMode) list.classList.add('select-mode');
            else list.classList.remove('select-mode');
        }

        function handleCardClick(event, index) {
            if (selectMode && event.target.type !== 'checkbox') {
                toggleItemSelection(event, filteredData[index].id);
            } else if (event.target.type !== 'checkbox') {
                selectItem(index);
            }
        }

        function toggleSelectMode() {
            selectMode = !selectMode;
            const btn = document.getElementById('selectBtn');
            const list = document.getElementById('itemList');

            if (selectMode) {
                btn.classList.add('active');
                btn.innerHTML = '&#10003; Exit Sel.';
                list.classList.add('select-mode');
            } else {
                btn.classList.remove('active');
                btn.innerHTML = '&#9745; Select';
                list.classList.remove('select-mode');
                selectedItems.clear();
            }
            updateSelectionUI();
            renderList();
        }

        function toggleItemSelection(event, itemId) {
            event.stopPropagation();
            if (selectedItems.has(itemId)) selectedItems.delete(itemId);
            else selectedItems.add(itemId);
            updateSelectionUI();
            renderList();
        }

        function updateSelectionUI() {
            const count = selectedItems.size;
            document.getElementById('selectionCount').textContent = `${count} sel.`;
            document.getElementById('selectionCount').classList.toggle('visible', count > 0);
            document.getElementById('batchDeleteBtn').classList.toggle('visible', count > 0);
            document.getElementById('batchEditBtn').classList.toggle('visible', count > 0);
        }

        function selectItem(index) {
            currentIndex = index;
            const item = filteredData[index];
            const color = getCollectionColor(item.collection);

            document.querySelectorAll('.item-card').forEach((el, i) => el.classList.toggle('active', i === index));
            document.querySelector('.item-card.active')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

            const viewer = document.getElementById('imageViewer');
            viewer.innerHTML = `
                <img src="${item.image_path}?t=${Date.now()}" onerror="this.outerHTML='<div class=\\'no-image\\'><p>Immagine non trovata</p></div>'">
                <button class="nav-btn prev" onclick="navigate(-1)">&#8249;</button>
                <button class="nav-btn next" onclick="navigate(1)">&#8250;</button>
                <div class="rotate-btns">
                    <button class="rotate-btn" onclick="rotateImage(-90)" title="Rotate left">&#8634;</button>
                    <button class="rotate-btn" onclick="rotateImage(90)" title="Rotate right">&#8635;</button>
                    <button class="rotate-btn" onclick="rotateImage(180)" title="Flip">&#8693;</button>
                </div>
            `;

            const fields = [
                { key: 'id', label: 'ID' },
                { key: 'collection', label: 'Collezione', class: 'collection' },
                { key: 'decoration', label: 'Decorazione', class: 'decoration', editable: true },
                { key: 'vessel_type', label: 'Tipo Vaso', class: 'vessel_type', editable: true },
                { key: 'part_type', label: 'Parte', class: 'part_type', editable: true },
                { key: 'macro_period', label: 'Macro-Periodo', class: 'macro-period' },
                { key: 'period', label: 'Periodo', class: 'period', editable: true },
                { key: 'page_ref', label: 'Rif. PDF', class: 'page-ref', clickable: true },
                { key: 'figure_num', label: 'Figura' },
                { key: 'pottery_id', label: 'ID Ceramica' },
            ];

            document.getElementById('metadataGrid').innerHTML = fields
                .filter(f => item[f.key])
                .map(f => `
                    <div class="meta-item ${f.class || ''} ${f.editable ? 'editable' : ''}" style="--collection-color: ${color}">
                        <div class="label">${f.label}</div>
                        <div class="value" ${f.clickable ? `onclick="openPdfAtPage('${item[f.key]}', '${item.collection}')"` : ''}>
                            ${item[f.key] || '-'}
                        </div>
                    </div>
                `).join('');
        }

        function navigate(dir) {
            if (filteredData.length === 0) return;
            let idx = currentIndex + dir;
            if (idx < 0) idx = filteredData.length - 1;
            if (idx >= filteredData.length) idx = 0;
            selectItem(idx);
        }

        function rotateImage(degrees) {
            const item = filteredData[currentIndex];
            if (!item) return;

            fetch('/api/rotate-image', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: item.image_path, degrees: degrees })
            })
            .then(r => r.json())
            .then(result => {
                if (result.success) {
                    // Reload image
                    selectItem(currentIndex);
                } else {
                    alert('Rotation error: ' + (result.error || 'Unknown'));
                }
            })
            .catch(err => alert('Error: ' + err));
        }

        function openPdfAtPage(pageRef, collection) {
            const item = filteredData[currentIndex];
            const page = pageRef || item?.page_ref || '1';
            const coll = collection || item?.collection || '';
            fetch(`/api/open-pdf?page=${encodeURIComponent(page)}&collection=${encodeURIComponent(coll)}`);
        }

        function openEditModal() {
            if (filteredData.length === 0) return;
            const item = filteredData[currentIndex];
            document.getElementById('editDecoration').value = item.decoration || 'plain';
            document.getElementById('editVesselType').value = item.vessel_type || 'unknown';
            document.getElementById('editPartType').value = item.part_type || 'rim';
            document.getElementById('editPeriod').value = item.period || '';
            document.getElementById('editModal').classList.add('active');
        }

        function saveEdit() {
            const item = filteredData[currentIndex];
            const fields = {
                decoration: document.getElementById('editDecoration').value,
                vessel_type: document.getElementById('editVesselType').value,
                part_type: document.getElementById('editPartType').value,
                period: document.getElementById('editPeriod').value
            };

            fetch('/api/update-item', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id: item.id, fields: fields })
            })
            .then(r => r.json())
            .then(result => {
                if (result.success) {
                    // Update local data
                    Object.assign(item, fields);
                    item.macro_period = getMacroPeriod(item.period);
                    closeModal('editModal');
                    selectItem(currentIndex);
                    renderList();
                } else {
                    alert('Error: ' + (result.error || 'Unknown'));
                }
            })
            .catch(err => alert('Error: ' + err));
        }

        function getMacroPeriod(period) {
            if (!period) return '';
            const p = period.toLowerCase();
            if (p.includes('umm') || p.includes('hili') || p.includes('2700') || p.includes('2500')) return 'Umm an-Nar';
            if (p.includes('wadi') || p.includes('2000-1600')) return 'Wadi Suq';
            if (p.includes('bronze') || p.includes('fer') || p.includes('masafi') || p.includes('1600')) return 'Late Bronze Age';
            return '';
        }

        function openBatchEditModal() {
            if (selectedItems.size === 0) return;
            document.getElementById('batchCount').textContent = selectedItems.size;
            document.getElementById('batchDecoration').value = '';
            document.getElementById('batchVesselType').value = '';
            document.getElementById('batchPartType').value = '';
            document.getElementById('batchEditModal').classList.add('active');
        }

        function saveBatchEdit() {
            const fields = {};
            const dec = document.getElementById('batchDecoration').value;
            const vessel = document.getElementById('batchVesselType').value;
            const part = document.getElementById('batchPartType').value;

            if (dec) fields.decoration = dec;
            if (vessel) fields.vessel_type = vessel;
            if (part) fields.part_type = part;

            if (Object.keys(fields).length === 0) {
                alert('Select at least one field to change');
                return;
            }

            fetch('/api/update-batch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ids: Array.from(selectedItems), fields: fields })
            })
            .then(r => r.json())
            .then(result => {
                if (result.success) {
                    // Update local data
                    for (const id of selectedItems) {
                        const item = data.find(d => d.id === id);
                        if (item) Object.assign(item, fields);
                    }
                    closeModal('batchEditModal');
                    applyFilters();
                    alert(`Updated ${result.updated} items`);
                } else {
                    alert('Error: ' + (result.error || 'Unknown'));
                }
            })
            .catch(err => alert('Error: ' + err));
        }

        function confirmDelete() {
            if (filteredData.length === 0) return;
            const item = filteredData[currentIndex];
            document.getElementById('deleteMessage').innerHTML = `Eliminare <strong>${item.id}</strong>?`;
            document.getElementById('deleteModal').dataset.batch = '';
            document.getElementById('deleteModal').classList.add('active');
        }

        function confirmBatchDelete() {
            if (selectedItems.size === 0) return;
            document.getElementById('deleteMessage').innerHTML = `Eliminare <strong>${selectedItems.size} elementi</strong>?`;
            document.getElementById('deleteModal').dataset.batch = 'true';
            document.getElementById('deleteModal').classList.add('active');
        }

        function closeModal(id) {
            document.getElementById(id).classList.remove('active');
        }

        function executeDelete() {
            const modal = document.getElementById('deleteModal');
            const isBatch = modal.dataset.batch === 'true';
            closeModal('deleteModal');

            if (isBatch) {
                const items = Array.from(selectedItems).map(id => {
                    const item = data.find(d => d.id === id);
                    return { id: item.id, path: item.image_path };
                });

                fetch('/api/delete-batch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(items)
                })
                .then(r => r.json())
                .then(result => {
                    if (result.success) {
                        for (const id of selectedItems) {
                            const idx = data.findIndex(d => d.id === id);
                            if (idx > -1) data.splice(idx, 1);
                        }
                        selectedItems.clear();
                        updateSelectionUI();
                        applyFilters();
                    }
                });
            } else {
                const item = filteredData[currentIndex];
                fetch(`/api/delete-image?path=${encodeURIComponent(item.image_path)}&id=${encodeURIComponent(item.id)}`, {
                    method: 'DELETE'
                })
                .then(r => r.json())
                .then(result => {
                    if (result.success) {
                        const idx = data.findIndex(d => d.id === item.id);
                        if (idx > -1) data.splice(idx, 1);
                        applyFilters();
                    }
                });
            }
        }

        document.addEventListener('keydown', e => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
            if (e.key === 'ArrowLeft') navigate(-1);
            if (e.key === 'ArrowRight') navigate(1);
            if (e.key === 'p' || e.key === 'P') openPdfAtPage();
            if (e.key === 'e' || e.key === 'E') openEditModal();
            if (e.key === 'r' || e.key === 'R') rotateImage(90);
            if (e.key === 'Escape') {
                closeModal('deleteModal');
                closeModal('editModal');
                closeModal('batchEditModal');
            }
        });
    </script>
</body>
</html>
'''

def main():
    os.chdir(Path(__file__).parent)

    config = load_config()
    save_config(config)

    # Generate index.html for static file serving
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(HTML_CONTENT)

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║         CeramicaDatabase - Unified Viewer                        ║
╠══════════════════════════════════════════════════════════════════╣
║  Server: http://{HOST}:{PORT}                                        ║
║                                                                  ║
║  Default password: ceramica2024                                  ║
║                                                                  ║
║  Keyboard shortcuts:                                             ║
║    ← →   : Navigate                                              ║
║    P     : Open PDF                                              ║
║    E     : Edit item                                             ║
║    R     : Rotate image 90°                                      ║
║    Esc   : Close dialog                                          ║
║                                                                  ║
║  Press Ctrl+C to stop                                            ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    # Only open browser when running locally
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
