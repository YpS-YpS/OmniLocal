"""Parse an image with OmniParser + Qwen OCR and save a labeled visualization.

Usage:
    python parse_image.py <image_path> [--port 8099] [--no-qwen]

Examples:
    python parse_image.py screenshot.png
    python parse_image.py "C:/screenshots/game menu.png"
    python parse_image.py screenshot.png --no-qwen          # use PaddleOCR instead
    python parse_image.py screenshot.png --port 8000        # different server port
"""

import argparse
import base64
import os
import sys
import time

import requests
from PIL import Image, ImageDraw, ImageFont


def parse_and_label(image_path, port=8099, use_qwen=True):
    # Read image
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    # Send to server
    payload = {"base64_image": img_b64}
    if use_qwen:
        payload["use_qwen_ocr"] = True

    print(f"Parsing: {image_path}")
    start = time.time()
    resp = requests.post(f"http://localhost:{port}/parse/", json=payload, timeout=300)
    elapsed = time.time() - start

    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text[:300]}")
        sys.exit(1)

    result = resp.json()
    elems = result.get("parsed_content_list", [])
    latency = result.get("latency", 0)

    # Print results
    text_elems = [e for e in elems if e.get("type") == "text"]
    icon_elems = [e for e in elems if e.get("type") == "icon"]
    print(f"Latency: {elapsed:.1f}s | {len(elems)} elements ({len(text_elems)} text, {len(icon_elems)} icons)\n")

    for i, elem in enumerate(elems):
        content = elem.get("content", "") or ""
        etype = elem.get("type", "")
        bbox = elem.get("bbox", [])
        print(f"  [{i:3d}] {etype:5s} | bbox={[round(b,3) for b in bbox]} | {content[:80]}")

    # Draw labeled image
    orig = Image.open(image_path).convert("RGB")
    w, h = orig.size
    draw = ImageDraw.Draw(orig)

    # Scale font to image size
    font_size = max(12, min(int(h / 60), 24))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    for i, elem in enumerate(elems):
        bbox = elem["bbox"]
        x1, y1, x2, y2 = int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)
        content = elem.get("content", "") or ""
        etype = elem.get("type", "")

        color = (0, 200, 0) if etype == "text" else (50, 120, 255)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        label = f"[{i}] {content[:40]}"
        label_y = max(y1 - font_size - 4, 0)
        tb = draw.textbbox((x1, label_y), label, font=font)
        draw.rectangle([tb[0]-1, tb[1]-1, tb[2]+1, tb[3]+1], fill=color)
        draw.text((x1, label_y), label, fill="white", font=font)

    # Save output
    name, ext = os.path.splitext(image_path)
    out_path = f"{name}_parsed{ext}"
    orig.save(out_path)
    print(f"\nSaved: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse image with OmniParser + Qwen OCR")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--port", type=int, default=8099, help="Server port (default: 8099)")
    parser.add_argument("--no-qwen", action="store_true", help="Use PaddleOCR instead of Qwen")
    args = parser.parse_args()

    parse_and_label(args.image, port=args.port, use_qwen=not args.no_qwen)
