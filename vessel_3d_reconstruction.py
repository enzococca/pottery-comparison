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
    Advanced profile extraction using contour analysis.
    Better handles complex vessel shapes.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Dilate to connect nearby edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None

    # Combine all contours to find the vessel shape
    all_points = np.vstack([c for c in contours if len(c) > 10])

    # Find bounding box
    x, y, w, h = cv2.boundingRect(all_points)

    # Estimate center line
    center_x = x + w // 2

    # Sample the profile at regular intervals
    num_samples = min(h, 200)
    profile_points = []

    for i in range(num_samples):
        row = y + int(i * h / num_samples)

        # Find rightmost edge point at this row
        rightmost = center_x
        for point in all_points:
            px, py = point[0]
            if abs(py - row) < 3 and px > rightmost:
                rightmost = px

        if rightmost > center_x:
            profile_points.append({
                'y': i * h / num_samples,
                'r': rightmost - center_x,
                'y_norm': i / num_samples,
                'r_norm': (rightmost - center_x) / (w / 2) if w > 0 else 0
            })

    # Smooth the profile
    if len(profile_points) > 5:
        radii = [p['r'] for p in profile_points]
        # Simple moving average
        window = 5
        smoothed = []
        for i in range(len(radii)):
            start = max(0, i - window // 2)
            end = min(len(radii), i + window // 2 + 1)
            smoothed.append(np.mean(radii[start:end]))

        for i, p in enumerate(profile_points):
            p['r_smooth'] = smoothed[i]

    if debug:
        debug_img = img.copy()
        cv2.line(debug_img, (center_x, y), (center_x, y + h), (0, 255, 0), 2)
        for p in profile_points:
            r = p.get('r_smooth', p['r'])
            px = int(center_x + r)
            py = int(y + p['y'])
            cv2.circle(debug_img, (px, py), 2, (0, 0, 255), -1)
        cv2.imwrite('/tmp/profile_debug_advanced.png', debug_img)

    return {
        'points': profile_points,
        'width': w,
        'height': h,
        'center_x': center_x,
        'bbox': (x, y, w, h)
    }


def generate_3d_mesh(profile_data, segments=64, scale=1.0):
    """
    Generate a 3D mesh by revolving the profile around the Y axis.

    Args:
        profile_data: dict with 'points' list containing y, r values
        segments: number of segments around the circumference
        scale: scale factor for the output mesh

    Returns:
        dict with 'vertices', 'faces', 'normals' for the mesh
    """
    points = profile_data['points']
    if not points:
        return None

    vertices = []
    faces = []
    uvs = []

    # Generate vertices by revolving profile
    for i, p in enumerate(points):
        y = p['y'] * scale
        r = p.get('r_smooth', p['r']) * scale
        v = p['y_norm']  # UV v coordinate

        for j in range(segments):
            angle = 2 * np.pi * j / segments
            x = r * np.cos(angle)
            z = r * np.sin(angle)

            vertices.append([x, -y, z])  # -y to flip (top of vessel at top)

            u = j / segments  # UV u coordinate
            uvs.append([u, v])

    # Generate faces (quads split into triangles)
    for i in range(len(points) - 1):
        for j in range(segments):
            # Current ring indices
            curr = i * segments + j
            next_j = i * segments + (j + 1) % segments
            # Next ring indices
            below = (i + 1) * segments + j
            below_next = (i + 1) * segments + (j + 1) % segments

            # Two triangles per quad
            faces.append([curr, below, next_j])
            faces.append([next_j, below, below_next])

    # Calculate normals
    vertices_np = np.array(vertices)
    normals = []

    for v in vertices:
        # For a surface of revolution, normal points outward from axis
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
        'segments': segments
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
        'height': texture_colored.shape[0]
    }


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


def reconstruct_vessel(image_path, output_dir=None, debug=False, with_decoration=True):
    """
    Main function to reconstruct a 3D vessel from a profile drawing.

    Returns: dict with mesh data and file paths
    """
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
    if with_decoration:
        print("Extracting decoration...")
        decoration = extract_decoration(image_path, profile, debug=debug)
        if decoration:
            print(f"  Decoration: {decoration['width']}x{decoration['height']}, has_decoration={decoration['has_decoration']}")
        else:
            print("  No decoration extracted")

    # Export to GLB (with or without texture)
    glb_path = os.path.join(output_dir, f"{base_name}.glb")
    if decoration and decoration.get('texture') is not None:
        glb_textured_path = os.path.join(output_dir, f"{base_name}_textured.glb")
        try:
            export_glb_with_texture(mesh, decoration, glb_textured_path)
            print(f"  Exported textured GLB: {glb_textured_path}")
        except Exception as e:
            print(f"  Failed to export textured GLB: {e}")
            glb_textured_path = None
    else:
        glb_textured_path = None

    # Also export basic GLB
    export_glb(mesh, glb_path)
    print(f"  Exported GLB: {glb_path}")

    return {
        'profile': profile,
        'mesh': mesh,
        'obj_path': obj_path,
        'glb_path': glb_path,
        'glb_textured_path': glb_textured_path,
        'decoration': decoration
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
