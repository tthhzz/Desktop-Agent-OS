"""Screen Perception MCP Server — intelligent screen understanding with OCR + SOM.

Provides these tools:
- screen_capture_and_parse — Capture screenshot, detect UI elements via OCR, generate SOM-annotated image
- screen_find_element — Find a specific UI element by description
- screen_read_text — OCR-read all text on screen

Uses pytesseract for OCR (lightweight, no GPU needed). Falls back gracefully
if tesseract is not installed.
"""

import base64
import io
import json
from typing import Optional

import pyautogui
from PIL import Image, ImageDraw, ImageFont
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("screen-perception")

# Element colors for SOM annotation
SOM_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
    "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
    "#BB8FCE", "#85C1E9", "#F0B27A", "#82E0AA",
]

# VLM-optimized image encoding settings
_VLM_MAX_WIDTH = 1280
_VLM_JPEG_QUALITY = 75


def _encode_image_for_vlm(img: Image.Image) -> str:
    """Encode a PIL Image as JPEG base64, optimized for VLM consumption.

    Resizes to max 1280px wide, converts to JPEG at 75% quality.
    This reduces a typical 1920x1080 PNG (2-5MB) to ~200-400KB JPEG.
    """
    # Resize if wider than max
    if img.width > _VLM_MAX_WIDTH:
        ratio = _VLM_MAX_WIDTH / img.width
        new_h = int(img.height * ratio)
        img = img.resize((_VLM_MAX_WIDTH, new_h), Image.LANCZOS)

    # Convert RGBA to RGB (JPEG doesn't support alpha)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_VLM_JPEG_QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _run_ocr(img: Image.Image, confidence: float = 50.0) -> list:
    """Run OCR on an image and return detected text elements.

    Returns list of dicts: [{"text": str, "x": int, "y": int, "w": int, "h": int, "conf": float}]
    """
    try:
        import pytesseract
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        elements = []
        for i, text in enumerate(data["text"]):
            if not text or not text.strip():
                continue
            conf = float(data["conf"][i])
            if conf < confidence:
                continue

            # Merge very close text into lines (skip single chars)
            if len(text.strip()) < 2:
                continue

            elements.append({
                "id": i + 1,
                "text": text.strip(),
                "type": "text",
                "x": data["left"][i],
                "y": data["top"][i],
                "w": data["width"][i],
                "h": data["height"][i],
                "conf": conf / 100.0,
            })

        return elements

    except ImportError:
        return []
    except Exception:
        return []


def _annotate_som(img: Image.Image, elements: list) -> Image.Image:
    """Draw Set-of-Mark annotations on the image.

    Each detected element gets a numbered box with a color-coded label.
    """
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for i, elem in enumerate(elements):
        color = SOM_COLORS[i % len(SOM_COLORS)]
        x, y, w, h = elem["x"], elem["y"], elem["w"], elem["h"]

        # Draw bounding box
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)

        # Draw label badge
        label = f"#{elem['id']}"
        bbox = font.getbbox(label) if hasattr(font, 'getbbox') else (0, 0, 30, 14)
        label_w = bbox[2] - bbox[0] + 6
        label_h = bbox[3] - bbox[1] + 4

        # Position label above or below box
        label_y = y - label_h - 2 if y > label_h + 2 else y + h + 2
        draw.rectangle(
            [x, label_y, x + label_w, label_y + label_h],
            fill=color,
        )
        draw.text((x + 3, label_y + 1), label, fill="white", font=font)

    return annotated


def _merge_nearby_elements(elements: list, gap: int = 10) -> list:
    """Merge text elements that are close together into single lines.

    This improves readability of the element list by combining
    fragmented OCR results.
    """
    if not elements:
        return []

    # Sort by Y then X
    sorted_elems = sorted(elements, key=lambda e: (e["y"], e["x"]))
    merged = []
    current_line = [sorted_elems[0]]

    for elem in sorted_elems[1:]:
        prev = current_line[-1]
        # Same line if Y overlap > 50% and X gap is small
        y_overlap = min(prev["y"] + prev["h"], elem["y"] + elem["h"]) - max(prev["y"], elem["y"])
        min_h = min(prev["h"], elem["h"])
        same_line = y_overlap > min_h * 0.5 and abs(elem["x"] - (prev["x"] + prev["w"])) < gap * 3

        if same_line:
            current_line.append(elem)
        else:
            # Merge current line
            if len(current_line) > 1:
                merged.append(_merge_line(current_line))
            else:
                merged.append(current_line[0])
            current_line = [elem]

    # Don't forget the last line
    if current_line:
        if len(current_line) > 1:
            merged.append(_merge_line(current_line))
        else:
            merged.append(current_line[0])

    # Re-number
    for i, elem in enumerate(merged):
        elem["id"] = i + 1

    return merged


def _merge_line(line_elems: list) -> dict:
    """Merge a list of elements on the same line into one."""
    text = " ".join(e["text"] for e in line_elems)
    x = min(e["x"] for e in line_elems)
    y = min(e["y"] for e in line_elems)
    right = max(e["x"] + e["w"] for e in line_elems)
    bottom = max(e["y"] + e["h"] for e in line_elems)
    conf = max(e["conf"] for e in line_elems)

    return {
        "id": 0,  # Will be re-numbered
        "text": text,
        "type": "text",
        "x": x,
        "y": y,
        "w": right - x,
        "h": bottom - y,
        "conf": conf,
    }


@mcp.tool()
def screen_capture_and_parse(
    confidence: float = 0.6,
    merge_nearby: bool = True,
    annotate: bool = True,
) -> str:
    """Capture the screen, detect UI elements via OCR, and return a SOM-annotated screenshot.

    This is the primary tool for screen understanding. It:
    1. Takes a screenshot
    2. Runs OCR to detect text elements
    3. Generates a Set-of-Mark (SOM) annotated image with numbered boxes
    4. Returns both the annotated image and a structured element list

    The LLM can then reference elements by ID number (e.g., "click #3").

    Args:
        confidence: Minimum OCR confidence (0.0-1.0). Default: 0.6.
        merge_nearby: Whether to merge close text elements into lines. Default: True.
        annotate: Whether to return the annotated image. Default: True.

    Returns:
        JSON with "elements" (list of detected UI elements) and optional "annotated_image" (base64 PNG).
    """
    img = pyautogui.screenshot()

    # Run OCR
    elements = _run_ocr(img, confidence=confidence * 100)

    # Merge nearby elements for cleaner output
    if merge_nearby:
        elements = _merge_nearby_elements(elements, gap=15)

    # Generate SOM annotation (VLM-optimized JPEG)
    annotated_b64 = None
    if annotate and elements:
        annotated_img = _annotate_som(img, elements)
        annotated_b64 = _encode_image_for_vlm(annotated_img)

    result = {
        "screen_size": f"{img.width}x{img.height}",
        "element_count": len(elements),
        "elements": elements[:50],  # Cap at 50 for efficiency
        "annotated_image": annotated_b64,
    }

    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def screen_find_element(
    description: str,
    confidence: float = 0.5,
) -> str:
    """Find a UI element on the screen by its text description.

    Uses OCR to detect all text elements, then searches for the
    one matching the description.

    Args:
        description: What to search for (e.g., "OK button", "File menu", "search box").
        confidence: Minimum OCR confidence (0.0-1.0). Default: 0.5.

    Returns:
        JSON with the matching element's position and ID, or "NOT_FOUND".
    """
    img = pyautogui.screenshot()
    elements = _run_ocr(img, confidence=confidence * 100)
    elements = _merge_nearby_elements(elements)

    desc_lower = description.lower()
    matches = []

    for elem in elements:
        text_lower = elem["text"].lower()
        # Check for exact or partial match
        if desc_lower in text_lower or text_lower in desc_lower:
            matches.append(elem)

    if matches:
        # Sort by confidence and text length (prefer more specific matches)
        matches.sort(key=lambda e: (len(e["text"]) * e["conf"]), reverse=True)
        best = matches[0]
        return json.dumps({
            "found": True,
            "element": best,
            "center": {"x": best["x"] + best["w"] // 2, "y": best["y"] + best["h"] // 2},
            "alternatives": [
                {"text": m["text"], "id": m["id"]} for m in matches[1:4]
            ],
        }, ensure_ascii=False)

    return json.dumps({"found": False, "description": description}, ensure_ascii=False)


@mcp.tool()
def screen_read_text(
    region: Optional[str] = None,
    confidence: float = 0.5,
) -> str:
    """Read all visible text on the screen using OCR.

    Args:
        region: Optional region as "left,top,width,height". If omitted, reads the full screen.
        confidence: Minimum OCR confidence (0.0-1.0). Default: 0.5.

    Returns:
        Extracted text with position information.
    """
    if region:
        try:
            parts = [int(x.strip()) for x in region.split(",")]
            if len(parts) == 4:
                img = pyautogui.screenshot(region=tuple(parts))
            else:
                img = pyautogui.screenshot()
        except (ValueError, IndexError):
            img = pyautogui.screenshot()
    else:
        img = pyautogui.screenshot()

    elements = _run_ocr(img, confidence=confidence * 100)
    elements = _merge_nearby_elements(elements)

    # Build readable text output
    lines = []
    for elem in elements:
        text = elem["text"]
        x, y = elem["x"], elem["y"]
        lines.append(f"[#{elem['id']} @({x},{y})] {text}")

    if not lines:
        return "No text detected on screen."

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
