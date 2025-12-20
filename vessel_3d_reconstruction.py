#!/usr/bin/env python3
"""
3D Vessel Reconstruction from Archaeological Profile Drawings

Takes a 2D profile drawing of a ceramic vessel and generates a 3D model
by revolving the profile around the central axis.

Features:
1. Profile extraction from drawing using edge detection
2. 3D mesh generation by revolution
3. Decoration texture mapping (if decoration visible)
4. Export to OBJ/GLB format for web viewing
"""

import os
import cv2
import numpy as np
from pathlib import Path
import json
import base64
import requests

# Global API key storage (can be set via environment or directly)
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')


def set_api_key(key):
    """Set the Anthropic API key."""
    global ANTHROPIC_API_KEY
    ANTHROPIC_API_KEY = key


def analyze_decoration_with_ai(image_path, profile_data, debug=False):
    """
    Use Claude AI to analyze and describe the decoration pattern on the vessel.

    Args:
        image_path: Path to the vessel profile image
        profile_data: Profile extraction data with bbox info

    Returns:
        dict with AI analysis results and enhanced decoration data
    """
    global ANTHROPIC_API_KEY

    if not ANTHROPIC_API_KEY:
        print("  No API key set, skipping AI analysis")
        return None

    # Read and prepare image
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Get the decoration region (right side of vessel)
    x, y, w, h = profile_data['bbox']
    center_x = profile_data['center_x']

    # Extract right half where decorations are shown
    height, width = img.shape[:2]
    dec_x1 = center_x
    dec_x2 = min(x + w + 30, width)
    dec_y1 = max(y - 20, 0)
    dec_y2 = min(y + h + 20, height)

    decoration_region = img[dec_y1:dec_y2, dec_x1:dec_x2]

    # Encode image to base64
    _, buffer = cv2.imencode('.png', decoration_region)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Call Claude API
    try:
        print("  Calling Claude AI for decoration analysis...")

        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': ANTHROPIC_API_KEY,
                'anthropic-version': '2023-06-01',
                'content-type': 'application/json'
            },
            json={
                'model': 'claude-sonnet-4-20250514',
                'max_tokens': 1024,
                'messages': [{
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image',
                            'source': {
                                'type': 'base64',
                                'media_type': 'image/png',
                                'data': img_base64
                            }
                        },
                        {
                            'type': 'text',
                            'text': '''Analyze this archaeological vessel drawing profile. Focus on the decoration patterns visible.

Please provide a structured JSON response with:
{
    "has_decoration": true/false,
    "decoration_type": "geometric/organic/figurative/abstract/none",
    "patterns": ["list of specific patterns like bands, lines, zigzag, spirals, dots, etc."],
    "technique": "incised/painted/relief/impressed/etc.",
    "coverage": "full/partial/rim_only/body_only",
    "bands": [
        {"position": "rim/neck/body/base", "pattern": "description", "height_percent": 0-100}
    ],
    "symmetry": "radial/bilateral/none",
    "description": "Brief archaeological description of the decoration style"
}

Respond ONLY with the JSON, no other text.'''
                        }
                    ]
                }]
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            content = result.get('content', [{}])[0].get('text', '{}')

            # Parse JSON response
            try:
                # Clean up response (remove markdown code blocks if present)
                content = content.strip()
                if content.startswith('```'):
                    content = content.split('\n', 1)[1]
                if content.endswith('```'):
                    content = content.rsplit('\n', 1)[0]
                content = content.strip()

                ai_analysis = json.loads(content)
                print(f"  AI Analysis: {ai_analysis.get('decoration_type', 'unknown')} decoration")

                if debug:
                    print(f"  Full AI response: {json.dumps(ai_analysis, indent=2)}")

                return {
                    'analysis': ai_analysis,
                    'decoration_region': decoration_region,
                    'success': True
                }
            except json.JSONDecodeError as e:
                print(f"  Failed to parse AI response: {e}")
                return {'raw_response': content, 'success': False}
        else:
            print(f"  API error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"  AI analysis error: {e}")
        return None


def generate_ai_decoration_texture(ai_analysis, profile_data, size=(512, 256)):
    """
    Generate a decoration texture based on AI analysis.
    Creates a procedural texture that represents the identified patterns.

    Args:
        ai_analysis: Result from analyze_decoration_with_ai
        profile_data: Profile data with dimensions
        size: Output texture size (width, height)

    Returns:
        numpy array with generated texture
    """
    if not ai_analysis or not ai_analysis.get('success'):
        return None

    analysis = ai_analysis.get('analysis', {})
    width, height = size

    # Create base ceramic color texture
    texture = np.ones((height, width, 3), dtype=np.uint8)
    base_color = [245, 235, 220]  # Light ceramic/terracotta
    texture[:, :] = base_color

    if not analysis.get('has_decoration', False):
        return texture

    patterns = analysis.get('patterns', [])
    bands = analysis.get('bands', [])
    decoration_type = analysis.get('decoration_type', 'unknown')

    print(f"  Generating texture for {decoration_type} decoration with patterns: {patterns}")

    # Draw horizontal bands if specified
    for band in bands:
        band_pos = band.get('position', 'body')
        band_pattern = band.get('pattern', 'lines')
        height_pct = band.get('height_percent', 50)

        # Calculate band position
        if band_pos == 'rim':
            y_start = 0
            y_end = int(height * 0.15)
        elif band_pos == 'neck':
            y_start = int(height * 0.1)
            y_end = int(height * 0.3)
        elif band_pos == 'body':
            y_start = int(height * 0.25)
            y_end = int(height * 0.75)
        elif band_pos == 'base':
            y_start = int(height * 0.8)
            y_end = height
        else:
            y_start = int(height * (height_pct - 10) / 100)
            y_end = int(height * (height_pct + 10) / 100)

        # Draw pattern in band
        draw_pattern_in_band(texture, y_start, y_end, band_pattern)

    # If no specific bands, draw based on patterns list
    if not bands and patterns:
        for i, pattern in enumerate(patterns[:3]):  # Max 3 patterns
            y_start = int(height * i / 3)
            y_end = int(height * (i + 1) / 3)
            draw_pattern_in_band(texture, y_start, y_end, pattern)

    return texture


def draw_pattern_in_band(texture, y_start, y_end, pattern_name):
    """Draw a specific pattern in a band of the texture."""
    height, width = texture.shape[:2]
    pattern_lower = pattern_name.lower()

    # Dark color for decoration lines
    dark = (40, 30, 20)  # Dark brown/black

    band_height = y_end - y_start

    if 'line' in pattern_lower or 'band' in pattern_lower:
        # Horizontal lines
        num_lines = max(2, band_height // 10)
        for i in range(num_lines):
            y = y_start + int(i * band_height / num_lines)
            cv2.line(texture, (0, y), (width, y), dark, 2)

    elif 'zigzag' in pattern_lower or 'chevron' in pattern_lower:
        # Zigzag pattern
        amplitude = band_height // 4
        y_mid = (y_start + y_end) // 2
        points = []
        for x in range(0, width, 15):
            y_offset = amplitude if (x // 15) % 2 == 0 else -amplitude
            points.append([x, y_mid + y_offset])
        if len(points) > 1:
            points = np.array(points, dtype=np.int32)
            cv2.polylines(texture, [points], False, dark, 2)

    elif 'spiral' in pattern_lower or 'curl' in pattern_lower:
        # Spiral patterns
        num_spirals = width // 50
        for i in range(num_spirals):
            cx = 25 + i * 50
            cy = (y_start + y_end) // 2
            for t in range(0, 270, 10):
                r = 3 + t / 30
                angle = np.radians(t)
                x = int(cx + r * np.cos(angle))
                y = int(cy + r * np.sin(angle))
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(texture, (x, y), 1, dark, -1)

    elif 'dot' in pattern_lower or 'point' in pattern_lower:
        # Dotted pattern
        spacing = 15
        for x in range(spacing, width - spacing, spacing):
            for y in range(y_start + spacing, y_end - spacing, spacing):
                cv2.circle(texture, (x, y), 3, dark, -1)

    elif 'crosshatch' in pattern_lower or 'hatch' in pattern_lower:
        # Crosshatch pattern
        spacing = 10
        for x in range(-height, width, spacing):
            cv2.line(texture, (x, y_start), (x + band_height, y_end), dark, 1)
            cv2.line(texture, (x + band_height, y_start), (x, y_end), dark, 1)

    elif 'wave' in pattern_lower or 'undulat' in pattern_lower:
        # Wave pattern
        y_mid = (y_start + y_end) // 2
        amplitude = band_height // 4
        points = []
        for x in range(width):
            y = int(y_mid + amplitude * np.sin(x * 0.1))
            points.append([x, y])
        if len(points) > 1:
            points = np.array(points, dtype=np.int32)
            cv2.polylines(texture, [points], False, dark, 2)

    elif 'triangle' in pattern_lower:
        # Triangle patterns
        for x in range(0, width, 30):
            pts = np.array([
                [x + 15, y_start + 5],
                [x + 5, y_end - 5],
                [x + 25, y_end - 5]
            ], dtype=np.int32)
            cv2.polylines(texture, [pts], True, dark, 2)

    elif 'circle' in pattern_lower or 'ring' in pattern_lower:
        # Concentric circles
        for x in range(30, width, 60):
            cy = (y_start + y_end) // 2
            for r in range(5, band_height // 2, 5):
                cv2.circle(texture, (x, cy), r, dark, 1)
    else:
        # Default: simple horizontal lines
        num_lines = max(2, band_height // 15)
        for i in range(num_lines):
            y = y_start + int(i * band_height / num_lines)
            cv2.line(texture, (0, y), (width, y), dark, 1)

def extract_profile(image_path, debug=False):
    """
    Extract the vessel profile from a drawing.
    Archaeological drawings show the vessel in section with:
    - Left half: solid black profile (the wall thickness)
    - Right half: outline showing the exterior surface
    - Center line: the axis of symmetry

    Returns: list of (x, y) points representing the outer profile
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to get binary image
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Find the largest contour (the vessel)
    largest = max(contours, key=cv2.contourArea)

    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest)

    # Find the center line (axis of symmetry)
    # Usually it's near the center of the bounding box
    center_x = x + w // 2

    # Extract the right side profile (exterior)
    # Go from top to bottom, finding the rightmost point at each y level
    profile_points = []

    for row in range(y, y + h):
        # Find all points at this row
        row_points = []
        for point in largest:
            px, py = point[0]
            if py == row and px >= center_x:
                row_points.append(px)

        if row_points:
            # Take the rightmost point
            rightmost = max(row_points)
            # Normalize: distance from center, and y position
            profile_points.append({
                'y': row - y,  # 0 = top of vessel
                'r': rightmost - center_x,  # radius from center
                'y_norm': (row - y) / h,  # normalized 0-1
                'r_norm': (rightmost - center_x) / (w / 2)  # normalized radius
            })

    if debug:
        # Draw debug image
        debug_img = img.copy()
        cv2.line(debug_img, (center_x, y), (center_x, y + h), (0, 255, 0), 2)
        for p in profile_points:
            px = center_x + p['r']
            py = y + p['y']
            cv2.circle(debug_img, (int(px), int(py)), 1, (0, 0, 255), -1)
        cv2.imwrite('/tmp/profile_debug.png', debug_img)

    return {
        'points': profile_points,
        'width': w,
        'height': h,
        'center_x': center_x,
        'bbox': (x, y, w, h)
    }


def extract_profile_advanced(image_path, debug=False):
    """
    Advanced profile extraction with wall thickness.
    Archaeological drawings show:
    - Left side: solid black section (wall cross-section)
    - Right side: exterior surface with decorations

    Extracts both inner and outer profiles to create a vessel with realistic wall thickness.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to get the solid black areas (wall section)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Edge detection for outer profile
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None

    # Combine all contours
    all_points = np.vstack([c for c in contours if len(c) > 10])

    # Find bounding box
    x, y, w, h = cv2.boundingRect(all_points)

    # Find the center line (axis of symmetry)
    center_x = x + w // 2

    # Sample profiles at regular intervals
    num_samples = min(h, 200)
    profile_points = []

    # First pass: find outer profile on right side
    for i in range(num_samples):
        row = y + int(i * h / num_samples)

        # Find the outer edge (rightmost point on right side)
        outer_r = 0
        for point in all_points:
            px, py = point[0]
            if abs(py - row) < 3 and px > center_x:
                r = px - center_x
                if r > outer_r:
                    outer_r = r

        if outer_r > 0:
            profile_points.append({
                'y': i * h / num_samples,
                'r_outer': outer_r,
                'r_inner': outer_r,  # Will be updated
                'y_norm': i / num_samples,
                'row': row
            })

    # Second pass: analyze left side to find wall thickness
    # The left side shows a solid black section representing the wall
    for p in profile_points:
        row = p['row']
        outer_r = p['r_outer']

        # Scan left side of center to find the black section width
        row_slice = binary[max(0, row-1):min(height, row+2), :]
        if row_slice.size > 0:
            row_avg = np.mean(row_slice, axis=0)

            # Find the leftmost edge of vessel on left side
            left_outer = center_x
            for px in range(center_x, max(0, x - 10), -1):
                if px < len(row_avg) and row_avg[px] > 200:  # White (outside vessel)
                    # Go right to find where vessel starts
                    for px2 in range(px, center_x):
                        if px2 < len(row_avg) and row_avg[px2] < 128:  # Black (wall)
                            left_outer = px2
                            break
                    break

            # Find inner edge of wall on left side
            left_inner = left_outer
            for px in range(left_outer, center_x):
                if px < len(row_avg) and row_avg[px] > 200:  # White (inside vessel)
                    left_inner = px
                    break

            # Wall thickness on left side
            wall_thickness = left_inner - left_outer

            # Apply same thickness to right side (symmetric vessel)
            if wall_thickness > 2:
                p['r_inner'] = max(1, outer_r - wall_thickness)
                p['thickness'] = wall_thickness
            else:
                # Default thickness as percentage of radius
                default_thickness = max(3, outer_r * 0.08)
                p['r_inner'] = max(1, outer_r - default_thickness)
                p['thickness'] = default_thickness

    # Remove row from points (not needed in output)
    for p in profile_points:
        if 'row' in p:
            del p['row']
        p['r_norm'] = p['r_outer'] / (w / 2) if w > 0 else 0

    # Smooth both profiles
    if len(profile_points) > 5:
        outer_radii = [p['r_outer'] for p in profile_points]
        inner_radii = [p['r_inner'] for p in profile_points]
        window = 5

        smoothed_outer = []
        smoothed_inner = []
        for i in range(len(outer_radii)):
            start = max(0, i - window // 2)
            end = min(len(outer_radii), i + window // 2 + 1)
            smoothed_outer.append(np.mean(outer_radii[start:end]))
            smoothed_inner.append(np.mean(inner_radii[start:end]))

        for i, p in enumerate(profile_points):
            p['r_outer_smooth'] = smoothed_outer[i]
            p['r_inner_smooth'] = smoothed_inner[i]
            p['r'] = p['r_outer']
            p['r_smooth'] = smoothed_outer[i]

    # Calculate average wall thickness
    thicknesses = [p.get('thickness', 0) for p in profile_points]
    avg_thickness = np.mean(thicknesses) if thicknesses else 0

    if debug:
        debug_img = img.copy()
        cv2.line(debug_img, (center_x, y), (center_x, y + h), (0, 255, 0), 2)
        for p in profile_points:
            py = int(y + p['y'])
            r_out = p.get('r_outer_smooth', p['r_outer'])
            r_in = p.get('r_inner_smooth', p['r_inner'])
            # Outer profile (red)
            cv2.circle(debug_img, (int(center_x + r_out), py), 2, (0, 0, 255), -1)
            # Inner profile (blue)
            cv2.circle(debug_img, (int(center_x + r_in), py), 2, (255, 0, 0), -1)
            # Also show on left side
            cv2.circle(debug_img, (int(center_x - r_out), py), 2, (0, 0, 255), -1)
            cv2.circle(debug_img, (int(center_x - r_in), py), 2, (255, 0, 0), -1)
        cv2.imwrite('/tmp/profile_debug_advanced.png', debug_img)
        print(f"  Average wall thickness: {avg_thickness:.1f} px")

    return {
        'points': profile_points,
        'width': w,
        'height': h,
        'center_x': center_x,
        'bbox': (x, y, w, h),
        'has_thickness': True,  # Always generate with thickness
        'avg_thickness': avg_thickness
    }


def generate_3d_mesh(profile_data, segments=64, scale=1.0):
    """
    Generate a 3D mesh by revolving the profile around the Y axis.
    Creates a vessel with wall thickness if profile data includes inner/outer radii.

    Args:
        profile_data: dict with 'points' list containing y, r_outer, r_inner values
        segments: number of segments around the circumference
        scale: scale factor for the output mesh

    Returns:
        dict with 'vertices', 'faces', 'normals' for the mesh
    """
    points = profile_data['points']
    if not points:
        return None

    has_thickness = profile_data.get('has_thickness', False)

    vertices = []
    faces = []
    uvs = []

    if has_thickness:
        # Generate vessel with wall thickness (outer + inner surfaces)
        print("  Generating mesh with wall thickness...")

        # OUTER SURFACE
        outer_start = 0
        for i, p in enumerate(points):
            y = p['y'] * scale
            r = p.get('r_outer_smooth', p.get('r_outer', p.get('r_smooth', p['r']))) * scale
            v = p['y_norm']

            for j in range(segments):
                angle = 2 * np.pi * j / segments
                x = r * np.cos(angle)
                z = r * np.sin(angle)
                vertices.append([x, -y, z])
                uvs.append([j / segments, v])

        # INNER SURFACE (reversed winding for correct normals)
        inner_start = len(vertices)
        for i, p in enumerate(points):
            y = p['y'] * scale
            r = p.get('r_inner_smooth', p.get('r_inner', p.get('r_smooth', p['r']) * 0.9)) * scale
            v = p['y_norm']

            for j in range(segments):
                angle = 2 * np.pi * j / segments
                x = r * np.cos(angle)
                z = r * np.sin(angle)
                vertices.append([x, -y, z])
                uvs.append([j / segments, v])

        # Outer surface faces
        for i in range(len(points) - 1):
            for j in range(segments):
                curr = outer_start + i * segments + j
                next_j = outer_start + i * segments + (j + 1) % segments
                below = outer_start + (i + 1) * segments + j
                below_next = outer_start + (i + 1) * segments + (j + 1) % segments
                faces.append([curr, below, next_j])
                faces.append([next_j, below, below_next])

        # Inner surface faces (reversed winding)
        for i in range(len(points) - 1):
            for j in range(segments):
                curr = inner_start + i * segments + j
                next_j = inner_start + i * segments + (j + 1) % segments
                below = inner_start + (i + 1) * segments + j
                below_next = inner_start + (i + 1) * segments + (j + 1) % segments
                faces.append([curr, next_j, below])
                faces.append([next_j, below_next, below])

        # Top rim (connect outer to inner at top) - creates the rim edge
        for j in range(segments):
            outer_curr = outer_start + j
            outer_next = outer_start + (j + 1) % segments
            inner_curr = inner_start + j
            inner_next = inner_start + (j + 1) % segments
            faces.append([outer_curr, inner_curr, outer_next])
            faces.append([outer_next, inner_curr, inner_next])

        # Bottom: check if vessel has a base or is open
        last_ring = len(points) - 1
        last_outer_r = points[-1].get('r_outer_smooth', points[-1].get('r_outer', points[-1].get('r', 0)))
        last_inner_r = points[-1].get('r_inner_smooth', points[-1].get('r_inner', last_outer_r * 0.9))

        # If bottom radius is very small, close with a disc (pointed base)
        if last_outer_r < 5:
            # Create a center point for the bottom
            center_idx = len(vertices)
            last_y = points[-1]['y'] * scale
            vertices.append([0, -last_y, 0])
            uvs.append([0.5, 1.0])

            # Connect outer ring to center
            for j in range(segments):
                curr = outer_start + last_ring * segments + j
                next_j = outer_start + last_ring * segments + (j + 1) % segments
                faces.append([curr, center_idx, next_j])
        else:
            # Connect outer to inner at bottom (creates bottom rim, not closed disc)
            for j in range(segments):
                outer_curr = outer_start + last_ring * segments + j
                outer_next = outer_start + last_ring * segments + (j + 1) % segments
                inner_curr = inner_start + last_ring * segments + j
                inner_next = inner_start + last_ring * segments + (j + 1) % segments
                faces.append([outer_curr, outer_next, inner_curr])
                faces.append([outer_next, inner_next, inner_curr])

    else:
        # Simple surface (no thickness) - original behavior
        for i, p in enumerate(points):
            y = p['y'] * scale
            r = p.get('r_smooth', p['r']) * scale
            v = p['y_norm']

            for j in range(segments):
                angle = 2 * np.pi * j / segments
                x = r * np.cos(angle)
                z = r * np.sin(angle)
                vertices.append([x, -y, z])
                uvs.append([j / segments, v])

        for i in range(len(points) - 1):
            for j in range(segments):
                curr = i * segments + j
                next_j = i * segments + (j + 1) % segments
                below = (i + 1) * segments + j
                below_next = (i + 1) * segments + (j + 1) % segments
                faces.append([curr, below, next_j])
                faces.append([next_j, below, below_next])

    # Calculate normals
    normals = []
    for v in vertices:
        nx, ny, nz = v[0], 0, v[2]
        length = np.sqrt(nx*nx + nz*nz)
        if length > 0:
            normals.append([nx/length, ny, nz/length])
        else:
            normals.append([0, 1, 0])

    return {
        'vertices': vertices,
        'faces': faces,
        'uvs': uvs,
        'normals': normals,
        'profile_points': len(points),
        'segments': segments,
        'has_thickness': has_thickness
    }


def export_obj(mesh_data, filename):
    """Export mesh to OBJ format."""
    with open(filename, 'w') as f:
        f.write("# 3D Vessel Reconstruction\n")
        f.write(f"# Vertices: {len(mesh_data['vertices'])}\n")
        f.write(f"# Faces: {len(mesh_data['faces'])}\n\n")

        # Vertices
        for v in mesh_data['vertices']:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Texture coordinates
        for uv in mesh_data['uvs']:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

        # Normals
        for n in mesh_data['normals']:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

        # Faces (OBJ is 1-indexed)
        f.write("\n")
        for face in mesh_data['faces']:
            # Format: f v/vt/vn v/vt/vn v/vt/vn
            f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} "
                   f"{face[1]+1}/{face[1]+1}/{face[1]+1} "
                   f"{face[2]+1}/{face[2]+1}/{face[2]+1}\n")

    return filename


def export_glb(mesh_data, filename):
    """Export mesh to GLB format (binary glTF) for web viewing."""
    try:
        import struct
    except:
        return None

    # This is a simplified GLB exporter
    # For production, use a library like pygltflib

    vertices = np.array(mesh_data['vertices'], dtype=np.float32)
    faces = np.array(mesh_data['faces'], dtype=np.uint32)

    # Create glTF JSON
    gltf = {
        "asset": {"version": "2.0", "generator": "VesselReconstruction"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": {"POSITION": 0},
                "indices": 1,
                "mode": 4  # TRIANGLES
            }]
        }],
        "accessors": [
            {  # Positions
                "bufferView": 0,
                "componentType": 5126,  # FLOAT
                "count": len(vertices),
                "type": "VEC3",
                "min": vertices.min(axis=0).tolist(),
                "max": vertices.max(axis=0).tolist()
            },
            {  # Indices
                "bufferView": 1,
                "componentType": 5125,  # UNSIGNED_INT
                "count": len(faces) * 3,
                "type": "SCALAR"
            }
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": vertices.nbytes},
            {"buffer": 0, "byteOffset": vertices.nbytes, "byteLength": faces.nbytes}
        ],
        "buffers": [{"byteLength": vertices.nbytes + faces.nbytes}]
    }

    # Binary data
    binary_data = vertices.tobytes() + faces.tobytes()

    # Pad to 4-byte alignment
    while len(binary_data) % 4 != 0:
        binary_data += b'\x00'

    json_str = json.dumps(gltf)
    while len(json_str) % 4 != 0:
        json_str += ' '
    json_bytes = json_str.encode('utf-8')

    # GLB header
    glb_data = b'glTF'  # Magic
    glb_data += struct.pack('<I', 2)  # Version
    glb_data += struct.pack('<I', 12 + 8 + len(json_bytes) + 8 + len(binary_data))  # Total length

    # JSON chunk
    glb_data += struct.pack('<I', len(json_bytes))  # Chunk length
    glb_data += b'JSON'  # Chunk type
    glb_data += json_bytes

    # Binary chunk
    glb_data += struct.pack('<I', len(binary_data))  # Chunk length
    glb_data += b'BIN\x00'  # Chunk type
    glb_data += binary_data

    with open(filename, 'wb') as f:
        f.write(glb_data)

    return filename


def extract_decoration(image_path, profile_data, debug=False):
    """
    Extract decoration pattern from the vessel drawing.
    In archaeological drawings, the right half typically shows
    the exterior surface with decorations.

    Returns: dict with texture data
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    height, width = img.shape[:2]
    x, y, w, h = profile_data['bbox']
    center_x = profile_data['center_x']

    # The decoration is on the right side of the center line
    dec_x1 = center_x
    dec_x2 = min(x + w + 20, width)
    dec_y1 = max(y - 10, 0)
    dec_y2 = min(y + h + 10, height)

    # Extract right half (decoration side)
    decoration_region = img[dec_y1:dec_y2, dec_x1:dec_x2].copy()

    if decoration_region.size == 0:
        return None

    # Create texture by mirroring for cylindrical mapping
    # Flip horizontally and concatenate for full cylinder
    dec_flipped = cv2.flip(decoration_region, 1)
    texture = np.hstack([decoration_region, dec_flipped])

    # Enhance decoration visibility
    gray = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)

    # Detect decorative patterns (lines, hatching, etc.)
    edges = cv2.Canny(gray, 30, 100)

    # Find contours that might be decorations
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    has_decoration = len([c for c in contours if cv2.contourArea(c) > 50]) > 5

    # Create a clean texture for 3D
    # Invert colors (black lines on white -> dark texture with light lines)
    texture_clean = cv2.bitwise_not(gray)

    # Apply ceramic color tint
    ceramic_color = np.array([108, 160, 201])  # BGR for terracotta-like
    texture_colored = np.zeros_like(texture)
    for i in range(3):
        texture_colored[:, :, i] = ((255 - texture_clean) / 255.0 * ceramic_color[i] +
                                    (texture_clean / 255.0 * 80)).astype(np.uint8)

    if debug:
        cv2.imwrite('/tmp/decoration_region.png', decoration_region)
        cv2.imwrite('/tmp/decoration_texture.png', texture_colored)

    return {
        'texture': texture_colored,
        'has_decoration': has_decoration,
        'width': texture_colored.shape[1],
        'height': texture_colored.shape[0],
        'raw_region': decoration_region  # Keep raw for AI enhancement
    }


def extract_decoration_lines(image_path, profile_data, debug=False):
    """
    Extract only the decoration lines from the vessel drawing,
    removing the vessel outline to create a clean repeatable pattern.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    height, width = img.shape[:2]
    x, y, w, h = profile_data['bbox']
    center_x = profile_data['center_x']

    # Extract decoration region (right side)
    dec_x1 = center_x + 5  # Skip a bit past center to avoid center line
    dec_x2 = min(x + w + 10, width)
    dec_y1 = max(y, 0)
    dec_y2 = min(y + h, height)

    decoration_region = img[dec_y1:dec_y2, dec_x1:dec_x2].copy()
    if decoration_region.size == 0:
        return None

    dec_h, dec_w = decoration_region.shape[:2]
    gray = cv2.cvtColor(decoration_region, cv2.COLOR_BGR2GRAY)

    # Threshold to get black lines
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find the outer contour (vessel outline) to remove it
    contours, _ = cv2.findContours(255 - binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create mask for decoration only (exclude vessel outline)
    decoration_mask = np.zeros_like(binary)
    inner_contours, _ = cv2.findContours(255 - binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in inner_contours:
        area = cv2.contourArea(cnt)
        # Skip very small noise and very large (vessel outline) contours
        if 20 < area < (dec_h * dec_w * 0.3):
            cv2.drawContours(decoration_mask, [cnt], -1, 255, -1)
            cv2.drawContours(decoration_mask, [cnt], -1, 255, 2)

    # Extract just the decoration lines
    decoration_lines = np.ones((dec_h, dec_w), dtype=np.uint8) * 255
    decoration_lines[decoration_mask > 0] = 0

    if debug:
        cv2.imwrite('/tmp/decoration_lines.png', decoration_lines)
        cv2.imwrite('/tmp/decoration_mask.png', decoration_mask)

    return {
        'lines': decoration_lines,
        'mask': decoration_mask,
        'region': decoration_region,
        'width': dec_w,
        'height': dec_h
    }


def create_cylindrical_texture_from_decoration(decoration_data, profile_data, texture_size=(1024, 512)):
    """
    Create a cylindrical texture by intelligently repeating the decoration pattern.
    The decoration from the drawing represents roughly 1/4 to 1/2 of the vessel surface.
    """
    if decoration_data is None:
        return None

    lines = decoration_data['lines']
    dec_h, dec_w = lines.shape[:2]
    tex_w, tex_h = texture_size

    # Create base texture (white/ceramic colored)
    texture = np.ones((tex_h, tex_w, 3), dtype=np.uint8) * 245  # Light ceramic

    # Scale decoration to fit texture height
    scale = tex_h / dec_h
    new_w = int(dec_w * scale)
    new_h = tex_h

    lines_scaled = cv2.resize(lines, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # The decoration typically shows ~90-180 degrees of the vessel
    # We need to tile it to cover 360 degrees
    # Assume decoration shows about 1/3 of circumference

    num_repeats = max(1, tex_w // new_w)
    remainder = tex_w % new_w if num_repeats > 0 else tex_w

    # Tile the decoration pattern
    for i in range(num_repeats):
        x_start = i * new_w
        x_end = min(x_start + new_w, tex_w)
        w_to_copy = x_end - x_start

        # Alternate: normal and mirrored for seamless tiling
        if i % 2 == 0:
            pattern = lines_scaled[:, :w_to_copy]
        else:
            pattern = cv2.flip(lines_scaled, 1)[:, :w_to_copy]

        # Apply decoration (black lines on white background)
        for c in range(3):
            texture[0:new_h, x_start:x_end, c] = np.where(
                pattern < 128,
                30,  # Dark brown/black for lines
                texture[0:new_h, x_start:x_end, c]
            )

    # Handle remainder
    if remainder > 0 and num_repeats > 0:
        x_start = num_repeats * new_w
        pattern = lines_scaled[:, :remainder]
        for c in range(3):
            texture[0:new_h, x_start:x_start+remainder, c] = np.where(
                pattern < 128,
                30,
                texture[0:new_h, x_start:x_start+remainder, c]
            )

    return texture


def enhance_decoration_with_ai(image_path, profile_data, api_key, debug=False):
    """
    Use Claude AI to analyze and enhance the decoration extraction.
    Returns an improved texture based on AI understanding of the pattern.
    """
    if not api_key:
        return None

    # First extract the raw decoration region
    img = cv2.imread(image_path)
    if img is None:
        return None

    x, y, w, h = profile_data['bbox']
    center_x = profile_data['center_x']
    height, width = img.shape[:2]

    # Get decoration region
    dec_x1 = center_x
    dec_x2 = min(x + w + 30, width)
    dec_y1 = max(y - 10, 0)
    dec_y2 = min(y + h + 10, height)

    decoration_region = img[dec_y1:dec_y2, dec_x1:dec_x2]

    # Encode for API
    _, buffer = cv2.imencode('.png', decoration_region)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    try:
        import sys
        print("  Analyzing decoration with Claude AI...", flush=True)

        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01',
                'content-type': 'application/json'
            },
            json={
                'model': 'claude-sonnet-4-20250514',
                'max_tokens': 2048,
                'messages': [{
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image',
                            'source': {
                                'type': 'base64',
                                'media_type': 'image/png',
                                'data': img_base64
                            }
                        },
                        {
                            'type': 'text',
                            'text': '''Sei un archeologo esperto in ceramica antica. Analizza questo disegno di profilo di un vaso archeologico.

IMPORTANTE: Concentrati SOLO sui MOTIVI DECORATIVI visibili sulla superficie del vaso (le linee disegnate che rappresentano la decorazione), NON sul contorno del vaso stesso.

Il disegno mostra tipicamente:
- Lato sinistro: sezione nera (spessore parete)
- Lato destro: superficie esterna con decorazioni

Analizza con precisione OGNI elemento decorativo e fornisci un JSON strutturato per la RICOSTRUZIONE 3D:

{
    "has_decoration": true/false,
    "decoration_type": "geometric/organic/figurative/abstract/mixed/none",
    "period_style": "stile/periodo se riconoscibile (es. Bronzo Antico, Bell Beaker, Appenninico, etc.)",
    "patterns": [
        {
            "type": "tipo specifico: horizontal_bands/parallel_lines/zigzag/chevron/herringbone/spiral/concentric_circles/dots/impressed_dots/comb_impressions/cord_impressions/waves/meander/triangles_filled/triangles_empty/hatching/cross_hatching/metope/triglyph/other",
            "position": "rim/lip/neck/shoulder/upper_body/mid_body/lower_body/base/handle",
            "height_percent_start": 0-100,
            "height_percent_end": 0-100,
            "line_thickness": "fine/medium/thick",
            "line_count": numero di linee se parallele,
            "spacing_mm": "stima spaziatura in mm se possibile",
            "spacing_type": "tight/regular/wide/irregular",
            "orientation": "horizontal/vertical/diagonal_left/diagonal_right/radial",
            "fill_pattern": "solid/hatched/dotted/empty",
            "repeat_count": "numero di ripetizioni visibili nel disegno",
            "continuity": "continuous/interrupted/alternating",
            "description_it": "descrizione dettagliata in italiano"
        }
    ],
    "band_structure": [
        {
            "band_name": "nome fascia (es. fascia orlo, fascia collo, registro principale)",
            "height_start": 0-100,
            "height_end": 0-100,
            "border_lines": numero di linee di bordo sopra/sotto,
            "main_motif": "motivo principale della fascia"
        }
    ],
    "repetition_module": "descrizione del modulo che si ripete (es. ogni 30 gradi, ogni 2cm)",
    "symmetry": "radial/bilateral/asymmetric",
    "technique": "incisione/impressione/pittura/applicazione_plastica/combinata",
    "tool_marks": "segni di strumento visibili (es. pettine a 4 denti, stecca, corda)",
    "complete_description": "Descrizione archeologica completa in italiano, inclusa interpretazione stilistica e possibile datazione/cultura di appartenenza. Descrivi come ricostruiresti la decorazione sulla parte non visibile del vaso."
}

IMPORTANTE per la ricostruzione:
- Specifica ESATTAMENTE dove inizia e finisce ogni pattern (percentuali precise)
- Indica se i pattern si ripetono e con quale ritmo
- Descrivi come continuerebbe la decorazione nella parte non visibile del vaso

Rispondi SOLO con JSON valido, senza altri commenti.'''
                        }
                    ]
                }]
            },
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            content = result.get('content', [{}])[0].get('text', '{}')

            # Parse JSON
            content = content.strip()
            if content.startswith('```'):
                content = content.split('\n', 1)[1]
            if content.endswith('```'):
                content = content.rsplit('\n', 1)[0]
            content = content.strip()

            analysis = json.loads(content)
            print(f"  AI analysis parsed successfully", flush=True)
            print(f"  - decoration_type: {analysis.get('decoration_type', 'N/A')}", flush=True)
            print(f"  - period_style: {analysis.get('period_style', 'N/A')}", flush=True)
            print(f"  - patterns found: {len(analysis.get('patterns', []))}", flush=True)
            for i, p in enumerate(analysis.get('patterns', [])[:3]):
                print(f"    [{i}] type={p.get('type', '?')}, position={p.get('position', '?')}", flush=True)

            # Now generate texture based on detailed analysis
            print("  Generating texture from AI analysis...", flush=True)
            texture = generate_texture_from_ai_analysis(analysis, decoration_region, profile_data)
            if texture is not None:
                print(f"  Texture generated: {texture.shape}", flush=True)
            else:
                print(f"  WARNING: Texture generation returned None!", flush=True)

            return {
                'analysis': analysis,
                'texture': texture,
                'success': True
            }

        else:
            print(f"  API error: {response.status_code}")
            print(f"  Response: {response.text[:500]}")
            return None

    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        print(f"  Content was: {content[:500]}")
        return None
    except Exception as e:
        import traceback
        print(f"  AI enhancement error: {e}")
        traceback.print_exc()
        return None


def generate_texture_from_ai_analysis(analysis, decoration_region, profile_data, size=(1024, 512)):
    """
    Generate a texture based on AI analysis.
    Uses the ACTUAL original decoration and tiles/mirrors it for reconstruction.
    The reconstruction should look realistic, not schematic.
    """
    tex_w, tex_h = size
    dec_h, dec_w = decoration_region.shape[:2]

    print(f"  Generating realistic texture from original decoration...", flush=True)

    # Start with ceramic base color
    texture = np.ones((tex_h, tex_w, 3), dtype=np.uint8)
    texture[:, :] = [245, 240, 230]  # Warm ceramic color (BGR)

    # Scale decoration to fit texture height while maintaining aspect ratio
    scale = tex_h / dec_h
    dec_scaled_w = int(dec_w * scale)
    dec_scaled_h = tex_h

    # Resize the original decoration (keep full color, high quality)
    decoration_scaled = cv2.resize(decoration_region, (dec_scaled_w, dec_scaled_h),
                                    interpolation=cv2.INTER_LANCZOS4)

    # Create mask for decoration (black lines on white background)
    gray = cv2.cvtColor(decoration_scaled, cv2.COLOR_BGR2GRAY)
    # Use adaptive threshold for better extraction of fine lines
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 5)

    # Clean up mask - remove noise
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Calculate how much of the original we have vs how much to reconstruct
    # Original takes approximately 1/3 of the circumference (what's visible in the drawing)
    original_width = min(dec_scaled_w, tex_w // 3)

    # === PLACE ORIGINAL DECORATION ===
    for y in range(tex_h):
        for x in range(min(original_width, dec_scaled_w)):
            if mask[y, x] > 100:
                # Use actual decoration color
                texture[y, x] = decoration_scaled[y, x]

    # === RECONSTRUCT THE REST BY TILING/MIRRORING ===
    # Create mirrored version for seamless tiling
    decoration_mirrored = cv2.flip(decoration_scaled, 1)
    mask_mirrored = cv2.flip(mask, 1)

    # Reconstruction starts after original
    recon_start = original_width

    # Tile the decoration across the remaining texture
    tile_x = 0
    for x in range(recon_start, tex_w):
        src_x = tile_x % dec_scaled_w
        # Alternate between mirrored and original for seamless tiling
        use_mirror = (tile_x // dec_scaled_w) % 2 == 0  # Start with mirrored

        for y in range(tex_h):
            if use_mirror:
                if mask_mirrored[y, src_x] > 100:
                    # Slightly lighter to indicate reconstruction
                    orig_color = decoration_mirrored[y, src_x].astype(np.int32)
                    new_color = np.clip(orig_color + 40, 0, 255).astype(np.uint8)
                    texture[y, x] = new_color
            else:
                if mask[y, src_x] > 100:
                    orig_color = decoration_scaled[y, src_x].astype(np.int32)
                    new_color = np.clip(orig_color + 40, 0, 255).astype(np.uint8)
                    texture[y, x] = new_color

        tile_x += 1

    # Add subtle separator line between original and reconstruction
    separator_color = (200, 190, 180)
    cv2.line(texture, (original_width, 0), (original_width, tex_h), separator_color, 1)

    # Add small labels (minimal, non-intrusive)
    label_color = (120, 110, 100)
    cv2.putText(texture, "orig.", (5, tex_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, label_color, 1)
    cv2.putText(texture, "ricostr.", (original_width + 5, tex_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, label_color, 1)

    return texture


def draw_dashed_line(img, pt1, pt2, color, thickness, dash_length, gap_length):
    """Draw a dashed line on an image."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Calculate line length and direction
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx*dx + dy*dy)

    if length == 0:
        return

    # Normalize direction
    dx /= length
    dy /= length

    # Draw dashes
    current_length = 0
    drawing = True

    while current_length < length:
        if drawing:
            # Calculate dash end point
            end_length = min(current_length + dash_length, length)
            start_x = int(x1 + dx * current_length)
            start_y = int(y1 + dy * current_length)
            end_x = int(x1 + dx * end_length)
            end_y = int(y1 + dy * end_length)
            cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)
            current_length = end_length
        else:
            current_length += gap_length
        drawing = not drawing


def export_glb_with_texture(mesh_data, texture_data, filename):
    """Export mesh to GLB format with texture."""
    try:
        import struct
    except:
        return None

    vertices = np.array(mesh_data['vertices'], dtype=np.float32)
    faces = np.array(mesh_data['faces'], dtype=np.uint32)
    uvs = np.array(mesh_data['uvs'], dtype=np.float32) if mesh_data.get('uvs') else None

    # Encode texture as PNG
    _, png_data = cv2.imencode('.png', texture_data['texture'])
    png_bytes = png_data.tobytes()

    # Pad to 4-byte alignment
    png_padding = (4 - len(png_bytes) % 4) % 4
    png_bytes_padded = png_bytes + b'\x00' * png_padding

    # Build buffer with vertices, faces, UVs, and image
    buffer_data = vertices.tobytes() + faces.tobytes()
    uv_offset = len(buffer_data)

    if uvs is not None:
        buffer_data += uvs.tobytes()

    image_offset = len(buffer_data)
    buffer_data += png_bytes_padded

    # Pad buffer to 4-byte alignment
    while len(buffer_data) % 4 != 0:
        buffer_data += b'\x00'

    # Create glTF JSON
    gltf = {
        "asset": {"version": "2.0", "generator": "VesselReconstruction"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": {"POSITION": 0},
                "indices": 1,
                "mode": 4,
                "material": 0
            }]
        }],
        "materials": [{
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.1,
                "roughnessFactor": 0.7
            }
        }],
        "textures": [{"sampler": 0, "source": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 33071}],
        "images": [{"bufferView": 3, "mimeType": "image/png"}],
        "accessors": [
            {  # Positions
                "bufferView": 0,
                "componentType": 5126,
                "count": len(vertices),
                "type": "VEC3",
                "min": vertices.min(axis=0).tolist(),
                "max": vertices.max(axis=0).tolist()
            },
            {  # Indices
                "bufferView": 1,
                "componentType": 5125,
                "count": len(faces) * 3,
                "type": "SCALAR"
            }
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": vertices.nbytes},
            {"buffer": 0, "byteOffset": vertices.nbytes, "byteLength": faces.nbytes},
            {"buffer": 0, "byteOffset": image_offset, "byteLength": len(png_bytes)}
        ],
        "buffers": [{"byteLength": len(buffer_data)}]
    }

    # Add UVs if available
    if uvs is not None:
        gltf["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"] = 2
        gltf["accessors"].append({
            "bufferView": 2,
            "componentType": 5126,
            "count": len(uvs),
            "type": "VEC2"
        })
        gltf["bufferViews"].insert(2, {
            "buffer": 0,
            "byteOffset": uv_offset,
            "byteLength": uvs.nbytes
        })
        # Update image bufferView index
        gltf["images"][0]["bufferView"] = 3

    json_str = json.dumps(gltf)
    while len(json_str) % 4 != 0:
        json_str += ' '
    json_bytes = json_str.encode('utf-8')

    # GLB header
    glb_data = b'glTF'
    glb_data += struct.pack('<I', 2)
    glb_data += struct.pack('<I', 12 + 8 + len(json_bytes) + 8 + len(buffer_data))

    # JSON chunk
    glb_data += struct.pack('<I', len(json_bytes))
    glb_data += b'JSON'
    glb_data += json_bytes

    # Binary chunk
    glb_data += struct.pack('<I', len(buffer_data))
    glb_data += b'BIN\x00'
    glb_data += buffer_data

    with open(filename, 'wb') as f:
        f.write(glb_data)

    return filename


def reconstruct_vessel(image_path, output_dir=None, debug=False, with_decoration=True, use_ai=False, api_key=None):
    """
    Main function to reconstruct a 3D vessel from a profile drawing.

    Args:
        image_path: Path to the profile drawing
        output_dir: Output directory for generated files
        debug: Enable debug output
        with_decoration: Extract decoration from image
        use_ai: Use Claude AI for decoration analysis
        api_key: Anthropic API key (optional, uses env var if not provided)

    Returns: dict with mesh data and file paths
    """
    # Set API key if provided
    if api_key:
        set_api_key(api_key)

    if output_dir is None:
        output_dir = '/tmp/vessel_3d'
    os.makedirs(output_dir, exist_ok=True)

    # Extract profile
    print(f"Extracting profile from: {image_path}")
    profile = extract_profile_advanced(image_path, debug=debug)

    if not profile or not profile['points']:
        print("  Failed to extract profile")
        return None

    print(f"  Found {len(profile['points'])} profile points")
    if profile.get('has_thickness'):
        print(f"  Wall thickness detected: {profile.get('avg_thickness', 0):.1f} px")

    # Generate 3D mesh
    print("Generating 3D mesh...")
    mesh = generate_3d_mesh(profile, segments=64, scale=0.1)

    if not mesh:
        print("  Failed to generate mesh")
        return None

    print(f"  Generated {len(mesh['vertices'])} vertices, {len(mesh['faces'])} faces")

    # Export to OBJ
    base_name = Path(image_path).stem
    obj_path = os.path.join(output_dir, f"{base_name}.obj")
    export_obj(mesh, obj_path)
    print(f"  Exported OBJ: {obj_path}")

    # Try to extract decoration
    decoration = None
    ai_analysis = None
    ai_texture = None

    if with_decoration:
        print("Extracting decoration...")
        decoration = extract_decoration(image_path, profile, debug=debug)
        if decoration:
            print(f"  Decoration: {decoration['width']}x{decoration['height']}, has_decoration={decoration['has_decoration']}")
        else:
            print("  No decoration extracted")

    # AI-based decoration analysis and enhancement
    if use_ai:
        print("Running AI decoration enhancement...", flush=True)
        # Use the new function that extracts and tiles actual decoration
        ai_result = enhance_decoration_with_ai(image_path, profile, api_key, debug=debug)
        print(f"  AI result: success={ai_result.get('success') if ai_result else 'None'}", flush=True)
        if ai_result and ai_result.get('success'):
            print("  AI enhancement successful!", flush=True)
            ai_analysis = ai_result.get('analysis')
            ai_texture = ai_result.get('texture')
            print(f"  ai_analysis keys: {list(ai_analysis.keys()) if ai_analysis else 'None'}", flush=True)
            print(f"  ai_texture shape: {ai_texture.shape if ai_texture is not None else 'None'}", flush=True)
            if ai_texture is not None:
                print(f"  Generated AI-enhanced texture: {ai_texture.shape}", flush=True)
        else:
            # Fallback: use basic AI analysis with procedural generation
            print("  Falling back to basic AI analysis...", flush=True)
            ai_basic = analyze_decoration_with_ai(image_path, profile, debug=debug)
            if ai_basic and ai_basic.get('success'):
                ai_analysis = ai_basic.get('analysis')
                ai_texture = generate_ai_decoration_texture(ai_basic, profile)
                if ai_texture is not None:
                    print(f"  Generated procedural AI texture: {ai_texture.shape}")

    # Export to GLB (with or without texture)
    glb_path = os.path.join(output_dir, f"{base_name}.glb")
    glb_textured_path = None
    glb_ai_path = None

    # Export with extracted decoration texture
    if decoration and decoration.get('texture') is not None:
        glb_textured_path = os.path.join(output_dir, f"{base_name}_textured.glb")
        try:
            export_glb_with_texture(mesh, decoration, glb_textured_path)
            print(f"  Exported textured GLB: {glb_textured_path}")
        except Exception as e:
            print(f"  Failed to export textured GLB: {e}")
            glb_textured_path = None

    # Export with AI-generated decoration texture
    print(f"  Checking ai_texture: {ai_texture is not None}", flush=True)
    if ai_texture is not None:
        glb_ai_path = os.path.join(output_dir, f"{base_name}_ai.glb")
        print(f"  Exporting AI GLB to: {glb_ai_path}", flush=True)
        try:
            ai_decoration_data = {
                'texture': ai_texture,
                'has_decoration': True,
                'width': ai_texture.shape[1],
                'height': ai_texture.shape[0]
            }
            export_glb_with_texture(mesh, ai_decoration_data, glb_ai_path)
            print(f"  Exported AI-textured GLB: {glb_ai_path}", flush=True)
        except Exception as e:
            import traceback
            print(f"  Failed to export AI-textured GLB: {e}", flush=True)
            traceback.print_exc()
            glb_ai_path = None

    # Also export basic GLB
    export_glb(mesh, glb_path)
    print(f"  Exported GLB: {glb_path}", flush=True)

    return {
        'profile': profile,
        'mesh': mesh,
        'obj_path': obj_path,
        'glb_path': glb_path,
        'glb_textured_path': glb_textured_path,
        'glb_ai_path': glb_ai_path,
        'decoration': decoration,
        'ai_analysis': ai_analysis,  # Already the analysis dict from enhance_decoration_with_ai
        'has_ai_decoration': ai_texture is not None
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = reconstruct_vessel(image_path, debug=True)
        if result:
            print(f"\nReconstruction complete!")
            print(f"OBJ: {result['obj_path']}")
            print(f"GLB: {result['glb_path']}")
    else:
        print("Usage: python vessel_3d_reconstruction.py <image_path>")
        print("\nExample: python vessel_3d_reconstruction.py Righetti/Fig.166/Righetti_90.png")
