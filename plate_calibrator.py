#!/usr/bin/env python3
"""
Plate Calibrator - Web interface to calibrate extracted PDF plates.
Allows measuring scale bars and saving pixels-per-cm for each plate.
"""

import os
import json
import sqlite3
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
from datetime import datetime

DB_PATH = "ceramica.db"
PORT = 8081
HOST = "127.0.0.1"

def get_db():
    return sqlite3.connect(DB_PATH)

def get_plates(collection=None, calibrated=None):
    """Get plates from database."""
    conn = get_db()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM plates WHERE 1=1"
    params = []

    if collection:
        query += " AND collection = ?"
        params.append(collection)

    if calibrated == 'yes':
        query += " AND pixels_per_cm IS NOT NULL"
    elif calibrated == 'no':
        query += " AND pixels_per_cm IS NULL"

    query += " ORDER BY collection, page_num"

    cursor.execute(query, params)
    plates = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return plates

def update_plate_calibration(plate_id, pixels_per_cm, scale_text=None):
    """Save calibration for a plate."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('''
        UPDATE plates
        SET pixels_per_cm = ?, scale_text = ?, calibrated_at = ?
        WHERE id = ?
    ''', (pixels_per_cm, scale_text, datetime.now().isoformat(), plate_id))

    conn.commit()
    conn.close()

def get_collections():
    """Get list of collections with plate counts."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT collection,
               COUNT(*) as total,
               SUM(CASE WHEN pixels_per_cm IS NOT NULL THEN 1 ELSE 0 END) as calibrated
        FROM plates
        GROUP BY collection
    ''')

    result = cursor.fetchall()
    conn.close()

    return [{'name': r[0], 'total': r[1], 'calibrated': r[2]} for r in result]

def apply_calibration_to_items(collection, page_num, pixels_per_cm):
    """Apply plate calibration to all items from that page."""
    conn = get_db()
    cursor = conn.cursor()

    # Find items from this page
    cursor.execute('''
        UPDATE items
        SET calibration_data = ?
        WHERE collection = ? AND page_ref LIKE ?
    ''', (json.dumps({'pixelsPerCm': pixels_per_cm, 'source': 'plate'}),
          collection, f'%{page_num}%'))

    updated = cursor.rowcount
    conn.commit()
    conn.close()

    return updated

class CalibratorHandler(SimpleHTTPRequestHandler):

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode('utf-8'))

    def send_html(self, content):
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)

        # Main page
        if parsed.path == '/' or parsed.path == '':
            self.send_html(get_calibrator_html())
            return

        # API: Get plates
        if parsed.path == '/api/plates':
            collection = query.get('collection', [None])[0]
            calibrated = query.get('calibrated', [None])[0]
            plates = get_plates(collection, calibrated)
            self.send_json(plates)
            return

        # API: Get collections
        if parsed.path == '/api/collections':
            collections = get_collections()
            self.send_json(collections)
            return

        # Serve plate images
        if parsed.path.startswith('/plates/'):
            return SimpleHTTPRequestHandler.do_GET(self)

        self.send_error(404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = json.loads(self.rfile.read(content_length).decode('utf-8')) if content_length > 0 else {}

        # Save calibration
        if parsed.path == '/api/calibrate':
            plate_id = post_data.get('plate_id')
            pixels_per_cm = post_data.get('pixels_per_cm')
            scale_text = post_data.get('scale_text')
            apply_to_items = post_data.get('apply_to_items', False)

            if plate_id and pixels_per_cm:
                update_plate_calibration(plate_id, pixels_per_cm, scale_text)

                updated_items = 0
                if apply_to_items:
                    # Get plate info
                    conn = get_db()
                    cursor = conn.cursor()
                    cursor.execute("SELECT collection, page_num FROM plates WHERE id = ?", (plate_id,))
                    row = cursor.fetchone()
                    conn.close()

                    if row:
                        updated_items = apply_calibration_to_items(row[0], row[1], pixels_per_cm)

                self.send_json({'success': True, 'updated_items': updated_items})
            else:
                self.send_json({'success': False, 'error': 'Missing data'})
            return

        self.send_error(404)


def get_calibrator_html():
    return '''<!DOCTYPE html>
<html>
<head>
    <title>Plate Calibrator</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            padding: 15px 20px;
            border-bottom: 1px solid #333;
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .header h1 { font-size: 1.3em; color: #4fc3f7; }
        .header select, .header button {
            padding: 8px 15px;
            border-radius: 5px;
            border: 1px solid #444;
            background: #0d0d1a;
            color: white;
            cursor: pointer;
        }
        .header button { background: #4fc3f7; color: #000; border: none; }
        .header button:hover { background: #29b6f6; }

        .stats {
            padding: 10px 20px;
            background: rgba(0,0,0,0.3);
            display: flex;
            gap: 20px;
            font-size: 0.9em;
        }
        .stat { color: #888; }
        .stat strong { color: #4fc3f7; }

        .container {
            display: flex;
            height: calc(100vh - 120px);
        }

        .sidebar {
            width: 250px;
            background: rgba(0,0,0,0.3);
            border-right: 1px solid #333;
            overflow-y: auto;
        }
        .plate-item {
            padding: 10px 15px;
            border-bottom: 1px solid #222;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .plate-item:hover { background: rgba(79, 195, 247, 0.1); }
        .plate-item.active { background: rgba(79, 195, 247, 0.2); border-left: 3px solid #4fc3f7; }
        .plate-item.calibrated { color: #4caf50; }
        .plate-item .page { font-weight: bold; }
        .plate-item .status { font-size: 0.8em; }

        .viewer {
            flex: 1;
            display: flex;
            flex-direction: column;
            position: relative;
        }
        .image-container {
            flex: 1;
            position: relative;
            overflow: auto;
            background: #0d0d1a;
        }
        .image-container img {
            max-width: 100%;
            display: block;
        }
        .measure-canvas {
            position: absolute;
            top: 0;
            left: 0;
            cursor: crosshair;
        }

        .calibration-panel {
            background: rgba(0,0,0,0.5);
            padding: 15px 20px;
            border-top: 1px solid #333;
        }
        .calibration-panel h3 { margin-bottom: 10px; color: #4fc3f7; }
        .calibration-row {
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }
        .calibration-row label { color: #888; }
        .calibration-row input, .calibration-row select {
            padding: 8px;
            border-radius: 5px;
            border: 1px solid #444;
            background: #0d0d1a;
            color: white;
            width: 100px;
        }
        .calibration-row button {
            padding: 8px 20px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
        }
        .btn-measure { background: #ff9800; color: #000; }
        .btn-measure.active { background: #f57c00; box-shadow: 0 0 10px rgba(255,152,0,0.5); }
        .btn-save { background: #4caf50; color: white; }
        .btn-save:hover { background: #43a047; }
        .btn-clear { background: #666; color: white; }

        .info { margin-top: 10px; color: #888; font-size: 0.9em; }
        .info strong { color: #4fc3f7; }

        .no-plate {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
            font-size: 1.2em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📐 Plate Calibrator</h1>
        <select id="collectionFilter" onchange="loadPlates()">
            <option value="">All Collections</option>
        </select>
        <select id="statusFilter" onchange="loadPlates()">
            <option value="">All Status</option>
            <option value="no">Not Calibrated</option>
            <option value="yes">Calibrated</option>
        </select>
        <button onclick="nextUncalibrated()">Next Uncalibrated →</button>
    </div>

    <div class="stats" id="stats"></div>

    <div class="container">
        <div class="sidebar" id="plateList"></div>

        <div class="viewer" id="viewer">
            <div class="no-plate">Select a plate from the list</div>
        </div>
    </div>

    <script>
        let plates = [];
        let currentPlate = null;
        let measureMode = false;
        let measurePoints = [];
        let canvas, ctx;

        async function loadCollections() {
            const res = await fetch('/api/collections');
            const collections = await res.json();

            const select = document.getElementById('collectionFilter');
            collections.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c.name;
                opt.textContent = `${c.name} (${c.calibrated}/${c.total})`;
                select.appendChild(opt);
            });

            // Stats
            const total = collections.reduce((a, c) => a + c.total, 0);
            const calibrated = collections.reduce((a, c) => a + c.calibrated, 0);
            document.getElementById('stats').innerHTML = `
                <span class="stat">Total: <strong>${total}</strong> plates</span>
                <span class="stat">Calibrated: <strong>${calibrated}</strong></span>
                <span class="stat">Remaining: <strong>${total - calibrated}</strong></span>
            `;
        }

        async function loadPlates() {
            const collection = document.getElementById('collectionFilter').value;
            const status = document.getElementById('statusFilter').value;

            let url = '/api/plates?';
            if (collection) url += `collection=${collection}&`;
            if (status) url += `calibrated=${status}&`;

            const res = await fetch(url);
            plates = await res.json();

            renderPlateList();
        }

        function renderPlateList() {
            const list = document.getElementById('plateList');
            list.innerHTML = plates.map((p, i) => `
                <div class="plate-item ${p.pixels_per_cm ? 'calibrated' : ''} ${currentPlate?.id === p.id ? 'active' : ''}"
                     onclick="selectPlate(${i})">
                    <span class="page">p. ${p.page_num}</span>
                    <span class="status">${p.pixels_per_cm ? '✓ ' + p.pixels_per_cm.toFixed(1) + ' px/cm' : '○'}</span>
                </div>
            `).join('');
        }

        function selectPlate(index) {
            currentPlate = plates[index];
            measurePoints = [];
            measureMode = false;

            const viewer = document.getElementById('viewer');
            viewer.innerHTML = `
                <div class="image-container" id="imageContainer">
                    <img src="${currentPlate.file_path}" id="plateImage" onload="initCanvas()">
                    <canvas class="measure-canvas" id="measureCanvas"></canvas>
                </div>
                <div class="calibration-panel">
                    <h3>Calibration - Page ${currentPlate.page_num} (${currentPlate.collection})</h3>
                    <div class="calibration-row">
                        <button class="btn-measure" id="measureBtn" onclick="toggleMeasure()">📏 Measure Scale Bar</button>
                        <label>Distance:</label>
                        <input type="number" id="distanceInput" value="10" step="0.5" min="0.1">
                        <select id="unitSelect">
                            <option value="cm">cm</option>
                            <option value="mm">mm</option>
                        </select>
                        <button class="btn-clear" onclick="clearMeasure()">Clear</button>
                        <button class="btn-save" onclick="saveCalibration()">💾 Save</button>
                    </div>
                    <div class="info" id="measureInfo">
                        ${currentPlate.pixels_per_cm ?
                            `<strong>Calibrated:</strong> ${currentPlate.pixels_per_cm.toFixed(2)} px/cm (${currentPlate.scale_text || ''})` :
                            'Click "Measure Scale Bar", then click two points on the scale bar'}
                    </div>
                </div>
            `;

            renderPlateList();
        }

        function initCanvas() {
            const img = document.getElementById('plateImage');
            canvas = document.getElementById('measureCanvas');
            ctx = canvas.getContext('2d');

            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            canvas.style.width = img.width + 'px';
            canvas.style.height = img.height + 'px';

            canvas.addEventListener('click', handleCanvasClick);
            canvas.addEventListener('mousemove', handleCanvasMove);
        }

        function toggleMeasure() {
            measureMode = !measureMode;
            measurePoints = [];
            const btn = document.getElementById('measureBtn');
            btn.classList.toggle('active', measureMode);

            if (measureMode) {
                document.getElementById('measureInfo').innerHTML = 'Click the <strong>FIRST</strong> point on the scale bar';
            }

            redraw();
        }

        function clearMeasure() {
            measurePoints = [];
            measureMode = false;
            document.getElementById('measureBtn').classList.remove('active');
            document.getElementById('measureInfo').innerHTML = 'Measurement cleared';
            redraw();
        }

        function handleCanvasClick(e) {
            if (!measureMode) return;

            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const x = (e.clientX - rect.left) * scaleX;
            const y = (e.clientY - rect.top) * scaleY;

            measurePoints.push({ x, y });
            redraw();

            if (measurePoints.length === 1) {
                document.getElementById('measureInfo').innerHTML = 'Click the <strong>SECOND</strong> point on the scale bar';
            } else if (measurePoints.length === 2) {
                const dx = measurePoints[1].x - measurePoints[0].x;
                const dy = measurePoints[1].y - measurePoints[0].y;
                const pixels = Math.sqrt(dx * dx + dy * dy);

                const distance = parseFloat(document.getElementById('distanceInput').value);
                const unit = document.getElementById('unitSelect').value;
                const distanceCm = unit === 'mm' ? distance / 10 : distance;

                const pxPerCm = pixels / distanceCm;

                document.getElementById('measureInfo').innerHTML =
                    `<strong>${pixels.toFixed(1)} pixels</strong> = ${distance} ${unit} → <strong>${pxPerCm.toFixed(2)} px/cm</strong>`;

                measureMode = false;
                document.getElementById('measureBtn').classList.remove('active');
            }
        }

        function handleCanvasMove(e) {
            if (!measureMode || measurePoints.length !== 1) return;

            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const x = (e.clientX - rect.left) * scaleX;
            const y = (e.clientY - rect.top) * scaleY;

            redraw();

            // Draw line to cursor
            ctx.beginPath();
            ctx.moveTo(measurePoints[0].x, measurePoints[0].y);
            ctx.lineTo(x, y);
            ctx.strokeStyle = '#ff9800';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        function redraw() {
            if (!ctx) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw points
            measurePoints.forEach((p, i) => {
                ctx.beginPath();
                ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
                ctx.fillStyle = '#ff9800';
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2;
                ctx.stroke();
            });

            // Draw line
            if (measurePoints.length === 2) {
                ctx.beginPath();
                ctx.moveTo(measurePoints[0].x, measurePoints[0].y);
                ctx.lineTo(measurePoints[1].x, measurePoints[1].y);
                ctx.strokeStyle = '#ff9800';
                ctx.lineWidth = 3;
                ctx.stroke();

                // Label
                const midX = (measurePoints[0].x + measurePoints[1].x) / 2;
                const midY = (measurePoints[0].y + measurePoints[1].y) / 2;
                const dx = measurePoints[1].x - measurePoints[0].x;
                const dy = measurePoints[1].y - measurePoints[0].y;
                const pixels = Math.sqrt(dx * dx + dy * dy);

                ctx.font = 'bold 16px Arial';
                ctx.fillStyle = '#fff';
                ctx.strokeStyle = '#000';
                ctx.lineWidth = 3;
                ctx.strokeText(pixels.toFixed(0) + ' px', midX + 10, midY - 10);
                ctx.fillText(pixels.toFixed(0) + ' px', midX + 10, midY - 10);
            }
        }

        async function saveCalibration() {
            if (measurePoints.length !== 2) {
                alert('Please measure the scale bar first');
                return;
            }

            const dx = measurePoints[1].x - measurePoints[0].x;
            const dy = measurePoints[1].y - measurePoints[0].y;
            const pixels = Math.sqrt(dx * dx + dy * dy);

            const distance = parseFloat(document.getElementById('distanceInput').value);
            const unit = document.getElementById('unitSelect').value;
            const distanceCm = unit === 'mm' ? distance / 10 : distance;
            const pxPerCm = pixels / distanceCm;

            const res = await fetch('/api/calibrate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    plate_id: currentPlate.id,
                    pixels_per_cm: pxPerCm,
                    scale_text: `${distance} ${unit}`,
                    apply_to_items: true
                })
            });

            const result = await res.json();
            if (result.success) {
                alert(`Saved! Calibration applied to ${result.updated_items} items.`);
                currentPlate.pixels_per_cm = pxPerCm;
                currentPlate.scale_text = `${distance} ${unit}`;
                loadCollections();
                loadPlates();
            }
        }

        function nextUncalibrated() {
            const uncalibrated = plates.find(p => !p.pixels_per_cm);
            if (uncalibrated) {
                const index = plates.indexOf(uncalibrated);
                selectPlate(index);
                document.querySelector('.plate-item.active')?.scrollIntoView({ behavior: 'smooth' });
            } else {
                alert('All plates in current view are calibrated!');
            }
        }

        // Init
        loadCollections();
        loadPlates();
    </script>
</body>
</html>
'''


def main():
    os.chdir(Path(__file__).parent)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              Plate Calibrator - Scale Measurement            ║
╠══════════════════════════════════════════════════════════════╣
║  Server: http://{HOST}:{PORT}                                     ║
║                                                              ║
║  Instructions:                                               ║
║  1. Select a plate from the list                             ║
║  2. Click "Measure Scale Bar"                                ║
║  3. Click two points on the scale bar in the image           ║
║  4. Enter the real distance (e.g., 10 cm)                    ║
║  5. Click "Save" to store calibration                        ║
║                                                              ║
║  Press Ctrl+C to stop                                        ║
╚══════════════════════════════════════════════════════════════╝
    """)

    import webbrowser
    webbrowser.open(f'http://{HOST}:{PORT}')

    server = HTTPServer((HOST, PORT), CalibratorHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == '__main__':
    main()
