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

# Configuration
PORT = int(os.environ.get('PORT', 8080))
HOST = os.environ.get('HOST', '0.0.0.0')
IS_RAILWAY = os.environ.get('RAILWAY_ENVIRONMENT') is not None

# Database
DB_FILE = "ceramica.db"
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

DEFAULT_CONFIG = {
    "collections": {
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
    db_path = Path(__file__).parent / DB_FILE
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize SQLite database"""
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

        # Open PDF (authenticated, macOS only)
        if parsed.path.startswith('/api/open-pdf'):
            page = query.get('page', ['1'])[0]
            collection = query.get('collection', [''])[0]

            page_match = re.search(r'p+\.\s*(\d+)', page)
            page_num = page_match.group(1) if page_match else '1'

            if sys.platform == 'darwin':
                config = load_config()
                pdf_path = config.get('collections', {}).get(collection, {}).get('pdf', '')
                if pdf_path:
                    base_path = Path(__file__).parent
                    full_pdf_path = str(base_path / pdf_path)
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
                    self.send_json({'success': True, 'page': page_num})
                else:
                    self.send_json({'error': 'PDF not found'})
            else:
                self.send_json({'error': 'Not supported on this platform'})
            return

        # Serve static files
        return SimpleHTTPRequestHandler.do_GET(self)

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
            <button class="rotate-btn" onclick="rotateImage(180)" title="Flip">&#8693;</button>
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
        .content {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}
        .image-viewer {{
            flex: 1;
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
        .delete-btn {{ background: linear-gradient(135deg, #f44336, #d32f2f); }}
        .select-btn {{ background: linear-gradient(135deg, #9c27b0, #7b1fa2); }}
        .select-btn.active {{ background: linear-gradient(135deg, #ff9800, #f57c00); }}
        .edit-btn {{ background: linear-gradient(135deg, #2196f3, #1976d2); }}
        .batch-btn {{ background: linear-gradient(135deg, #ff5722, #e64a19); display: none; }}
        .batch-btn.visible {{ display: inline-block; }}
        .logout-btn {{ background: linear-gradient(135deg, #607d8b, #455a64); }}
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
            z-index: 1000;
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
    </style>
</head>
<body>
    <div class="header">
        <h1>&#127994; CeramicaDatabase</h1>
        <div class="collection-tabs" id="collectionTabs"></div>
        <div class="header-buttons">
            <span class="user-badge">{'ADMIN' if is_admin else 'VIEWER'}</span>
            <span class="selection-count" id="selectionCount">0 sel.</span>
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
            periods.forEach(p => {{
                periodSelect.innerHTML += `<option value="${{p}}">${{p.substring(0,40)}}</option>`;
            }});

            // Populate other filters from vocabulary
            ['decoration', 'vessel_type', 'part_type'].forEach(field => {{
                const select = document.getElementById('filter' + field.charAt(0).toUpperCase() + field.slice(1).replace('_', ''));
                if (vocab[field]) {{
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
                     style="--collection-color: ${{getCollectionColor(item.collection)}}">
                    <input type="checkbox" class="item-checkbox"
                           ${{selectedItems.has(item.id) ? 'checked' : ''}}
                           onclick="toggleItemSelection(event, '${{item.id}}')">
                    <img src="${{item.image_path || ''}}" onerror="this.style.display='none'">
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
            viewer.innerHTML = `
                <img src="${{item.image_path}}?t=${{Date.now()}}" onerror="this.outerHTML='<div class=\\'no-image\\'><p>Image not found</p></div>'">
                <button class="nav-btn prev" onclick="navigate(-1)">&#8249;</button>
                <button class="nav-btn next" onclick="navigate(1)">&#8250;</button>
                {rotate_buttons}
            `;

            const fields = [
                {{ key: 'id', label: 'ID' }},
                {{ key: 'collection', label: 'Collection' }},
                {{ key: 'decoration', label: 'Decoration', class: 'decoration' }},
                {{ key: 'vessel_type', label: 'Vessel Type', class: 'vessel_type' }},
                {{ key: 'part_type', label: 'Part' }},
                {{ key: 'macro_period', label: 'Macro-Period' }},
                {{ key: 'period', label: 'Period', class: 'period' }},
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
        }}

        function navigate(dir) {{
            if (filteredData.length === 0) return;
            let idx = currentIndex + dir;
            if (idx < 0) idx = filteredData.length - 1;
            if (idx >= filteredData.length) idx = 0;
            selectItem(idx);
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
                if (result.success) selectItem(currentIndex);
                else alert('Rotation error: ' + (result.error || 'Unknown'));
            }})
            .catch(err => alert('Error: ' + err));
        }}

        function openPdfAtPage(pageRef, collection) {{
            const item = filteredData[currentIndex];
            const page = pageRef || item?.page_ref || '1';
            const coll = collection || item?.collection || '';
            fetch(`/api/open-pdf?page=${{encodeURIComponent(page)}}&collection=${{encodeURIComponent(coll)}}`);
        }}

        function openEditModal() {{
            if (!isAdmin || filteredData.length === 0) return;
            const item = filteredData[currentIndex];
            document.getElementById('editDecoration').value = item.decoration || '';
            document.getElementById('editVesselType').value = item.vessel_type || '';
            document.getElementById('editPartType').value = item.part_type || '';
            document.getElementById('editPeriod').value = item.period || '';
            document.getElementById('editModal').classList.add('active');
        }}

        function saveEdit() {{
            const item = filteredData[currentIndex];
            const fields = {{
                decoration: document.getElementById('editDecoration').value,
                vessel_type: document.getElementById('editVesselType').value,
                part_type: document.getElementById('editPartType').value,
                period: document.getElementById('editPeriod').value
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
            if (e.key === 'Escape') {{
                closeModal('deleteModal');
                closeModal('editModal');
                closeModal('batchEditModal');
            }}
        }});
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
    migrate_csv_to_db()

    # Load config
    config = load_config()

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║             CeramicaDatabase - Unified Viewer v2.0                   ║
╠══════════════════════════════════════════════════════════════════════╣
║  Server: http://{HOST}:{PORT}                                            ║
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
