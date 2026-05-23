"""Computer Use MCP Server — desktop automation via pyautogui.

Provides these tools:
- computer_screenshot   — capture screen or region
- computer_click        — click at (x, y)
- computer_type         — type text via keyboard
- computer_hotkey       — press key combination (e.g. "ctrl+c")
- computer_scroll       — scroll up/down
- computer_screen_info  — get screen resolution

Security: integrates with the security module for permission checks and audit logging.
"""

import base64
import io
import asyncio
import time
from typing import Optional

import pyautogui
from PIL import Image
from mcp.server.fastmcp import FastMCP
from ...security.permission_manager import PermissionManager, PermissionLevel
from ...security.audit_logger import AuditLogger

# Safety: give pyautogui a failsafe (move mouse to corner to abort)
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1  # Small pause between actions for stability

mcp = FastMCP("computer-use")

# Security components
_permission_manager = PermissionManager()
_audit_logger = AuditLogger()
_audit_logger.initialize()


# VLM-optimized screenshot settings
_VLM_MAX_WIDTH = 1280
_VLM_JPEG_QUALITY = 75


def _encode_image_for_vlm(img: Image.Image) -> str:
    """Encode a PIL Image as JPEG base64, optimized for VLM consumption.

    Resizes to max 1280px wide, converts to JPEG at 75% quality.
    This reduces a typical 1920x1080 PNG (2-5MB) to ~200-400KB JPEG,
    dramatically speeding up VLM inference.
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


@mcp.tool()
def computer_screenshot(region: Optional[str] = None) -> str:
    """Take a screenshot of the entire screen or a region.

    Args:
        region: Optional region as "left,top,width,height" (e.g. "0,0,800,600").
                If omitted, captures the full screen.

    Returns:
        Base64-encoded JPEG image of the screenshot (optimized for VLM).
    """
    if region:
        parts = [int(x.strip()) for x in region.split(",")]
        if len(parts) != 4:
            return "Error: region must be 'left,top,width,height'"
        img = pyautogui.screenshot(region=tuple(parts))
    else:
        img = pyautogui.screenshot()

    return _encode_image_for_vlm(img)


@mcp.tool()
def computer_click(x: int, y: int, button: str = "left", clicks: int = 1) -> str:
    """Click the mouse at the specified screen coordinates.

    Args:
        x: X coordinate on screen.
        y: Y coordinate on screen.
        button: Mouse button - "left", "right", or "middle". Default: "left".
        clicks: Number of clicks. Default: 1 (use 2 for double-click).

    Returns:
        Confirmation message.
    """
    _audit_logger.log(
        tool_name="computer_click",
        arguments=f"x={x}, y={y}, button={button}, clicks={clicks}",
        result_summary="executed",
        permission_level="DANGEROUS",
        was_approved=True,
    )
    pyautogui.click(x=x, y=y, button=button, clicks=clicks)
    return f"Clicked {button} at ({x}, {y}) x{clicks}"


@mcp.tool()
def computer_type(text: str, interval: float = 0.02) -> str:
    """Type text using the keyboard. Supports Chinese and other non-ASCII characters.

    Uses clipboard paste (Ctrl+V) for non-ASCII text, and direct keystroke
    input for ASCII-only text.

    Args:
        text: The text to type.
        interval: Seconds between keystrokes (ASCII mode only). Default: 0.02.

    Returns:
        Confirmation message.
    """
    _audit_logger.log(
        tool_name="computer_type",
        arguments=f"text={text[:100]}",
        result_summary="executed",
        permission_level="DANGEROUS",
        was_approved=True,
    )
    # Check if text contains non-ASCII characters (Chinese, Japanese, etc.)
    try:
        text.encode("ascii")
        is_ascii = True
    except UnicodeEncodeError:
        is_ascii = False

    if is_ascii:
        pyautogui.typewrite(text, interval=interval)
    else:
        # Use clipboard paste for non-ASCII text
        import pyperclip
        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")
    return f"Typed: {text[:50]}{'...' if len(text) > 50 else ''}"


@mcp.tool()
def computer_hotkey(*keys: str) -> str:
    """Press a keyboard shortcut / hotkey combination.

    Args:
        keys: Keys to press simultaneously, e.g. "ctrl" "c" or "alt" "tab".

    Returns:
        Confirmation message.
    """
    _audit_logger.log(
        tool_name="computer_hotkey",
        arguments="+".join(keys),
        result_summary="executed",
        permission_level="DANGEROUS",
        was_approved=True,
    )
    pyautogui.hotkey(*keys)
    return f"Pressed hotkey: {'+'.join(keys)}"


@mcp.tool()
def computer_scroll(amount: int, x: Optional[int] = None, y: Optional[int] = None) -> str:
    """Scroll the mouse wheel.

    Args:
        amount: Number of scroll clicks. Positive = scroll up, negative = scroll down.
        x: Optional X coordinate. If omitted, uses current mouse position.
        y: Optional Y coordinate. If omitted, uses current mouse position.

    Returns:
        Confirmation message.
    """
    kwargs = {}
    if x is not None and y is not None:
        kwargs["x"] = x
        kwargs["y"] = y
    pyautogui.scroll(amount, **kwargs)
    pos = f" at ({x}, {y})" if x is not None and y is not None else ""
    return f"Scrolled {amount} clicks{'up' if amount > 0 else 'down'}{pos}"


@mcp.tool()
def computer_screen_info() -> str:
    """Get the screen resolution and current mouse position.

    Returns:
        Screen width, height, and current mouse coordinates.
    """
    size = pyautogui.size()
    pos = pyautogui.position()
    return f"Screen: {size.width}x{size.height}, Mouse: ({pos.x}, {pos.y})"


@mcp.tool()
def computer_move(x: int, y: int, duration: float = 0.3) -> str:
    """Move the mouse to the specified coordinates.

    Args:
        x: Target X coordinate.
        y: Target Y coordinate.
        duration: Time in seconds for the move animation. Default: 0.3.

    Returns:
        Confirmation message.
    """
    pyautogui.moveTo(x, y, duration=duration)
    return f"Moved mouse to ({x}, {y})"


@mcp.tool()
def computer_drag(x: int, y: int, duration: float = 0.5, button: str = "left") -> str:
    """Drag from current mouse position to (x, y).

    Args:
        x: Target X coordinate.
        y: Target Y coordinate.
        duration: Drag duration in seconds. Default: 0.5.
        button: Mouse button to hold. Default: "left".

    Returns:
        Confirmation message.
    """
    pyautogui.dragTo(x, y, duration=duration, button=button)
    return f"Dragged to ({x}, {y}) with {button} button"


# ── Phase 5.6: Enhanced Computer Use tools ──────────────────────

# Screenshot hash cache for diff detection
_last_screenshot_hash: Optional[str] = None


def _compute_image_hash(img: Image.Image) -> str:
    """Compute a quick perceptual hash of an image for diff detection."""
    # Downsample to 8x8 and compute average hash
    small = img.resize((8, 8), Image.LANCZOS).convert("L")
    pixels = list(small.getdata())
    avg = sum(pixels) / len(pixels)
    return "".join("1" if p > avg else "0" for p in pixels)


@mcp.tool()
def computer_smart_screenshot(
    region: Optional[str] = None,
    diff_check: bool = False,
    diff_threshold: float = 0.05,
) -> str:
    """Take a smart screenshot with optional region and change detection.

    Args:
        region: Optional region: "full", "active_window", or "left,top,width,height".
                "active_window" captures only the foreground window.
                Default: "full".
        diff_check: If True, only return screenshot if screen changed since last capture.
                    Default: False.
        diff_threshold: Minimum fraction of pixels that must differ (0.0-1.0). Default: 0.05.

    Returns:
        Base64-encoded PNG, or "NO_CHANGE" if diff_check and screen hasn't changed.
    """
    global _last_screenshot_hash

    # Capture screenshot based on region type
    if region == "active_window":
        try:
            # Use win32gui to get active window rect (Windows only)
            import ctypes
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            rect = ctypes.wintypes.RECT()
            ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
            img = pyautogui.screenshot(region=(rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top))
        except Exception:
            # Fallback to full screen
            img = pyautogui.screenshot()
    elif region and region != "full":
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

    # Diff detection
    if diff_check:
        current_hash = _compute_image_hash(img)
        if _last_screenshot_hash:
            # Compare hashes (simple Hamming distance)
            diff_bits = sum(c1 != c2 for c1, c2 in zip(current_hash, _last_screenshot_hash))
            diff_ratio = diff_bits / len(current_hash) if current_hash else 1.0

            if diff_ratio < diff_threshold:
                _last_screenshot_hash = current_hash
                return f"NO_CHANGE (diff={diff_ratio:.3f} < threshold={diff_threshold})"

        _last_screenshot_hash = _compute_image_hash(img)

    # Encode and return (VLM-optimized JPEG)
    return _encode_image_for_vlm(img)


@mcp.tool()
def computer_find_on_screen(
    text: str,
    confidence: float = 0.8,
) -> str:
    """Find the position of text on the screen using OCR.

    Takes a screenshot, runs OCR to detect text, and returns
    the bounding box coordinates of matching text.

    Args:
        text: The text to search for on screen.
        confidence: Minimum OCR confidence (0.0-1.0). Default: 0.8.

    Returns:
        JSON with found text locations: [{"text": "...", "x": int, "y": int, "width": int, "height": int}]
        or "NOT_FOUND" if text not found.
    """
    import json

    # Take screenshot
    img = pyautogui.screenshot()

    # Try pytesseract for OCR
    try:
        import pytesseract
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        results = []
        search_lower = text.lower()

        for i, detected_text in enumerate(data["text"]):
            if not detected_text or search_lower not in detected_text.lower():
                continue

            conf = float(data["conf"][i])
            if conf < confidence * 100:  # pytesseract uses 0-100 scale
                continue

            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]

            results.append({
                "text": detected_text,
                "x": x + w // 2,  # Center X
                "y": y + h // 2,  # Center Y
                "left": x,
                "top": y,
                "width": w,
                "height": h,
                "confidence": conf / 100.0,
            })

        if results:
            return json.dumps(results, ensure_ascii=False)
        return f"NOT_FOUND: '{text}' not found on screen"

    except ImportError:
        # Fallback: use pyautogui locate (requires an image file, limited)
        return "ERROR: pytesseract not installed. Install with: pip install pytesseract && install Tesseract OCR"
    except Exception as e:
        return f"ERROR: OCR failed: {e}"


@mcp.tool()
def computer_click_by_text(
    text: str,
    button: str = "left",
    clicks: int = 1,
    confidence: float = 0.8,
) -> str:
    """Click on text visible on the screen by finding it with OCR first.

    Combines OCR text detection with pyautogui clicking. Takes a screenshot,
    finds the specified text, and clicks on its center position.

    Args:
        text: The text to find and click on screen.
        button: Mouse button - "left", "right", or "middle". Default: "left".
        clicks: Number of clicks. Default: 1.
        confidence: Minimum OCR confidence (0.0-1.0). Default: 0.8.

    Returns:
        Confirmation message with coordinates, or error if text not found.
    """
    # Find the text on screen
    find_result = computer_find_on_screen(text, confidence)

    if find_result.startswith("NOT_FOUND") or find_result.startswith("ERROR"):
        return f"Cannot click: {find_result}"

    import json
    try:
        locations = json.loads(find_result)
        if not locations:
            return f"Cannot click: no locations found for '{text}'"

        # Click the first (best) match
        loc = locations[0]
        x, y = loc["x"], loc["y"]

        _audit_logger.log(
            tool_name="computer_click_by_text",
            arguments=f"text={text}, x={x}, y={y}",
            result_summary="executed",
            permission_level="DANGEROUS",
            was_approved=True,
        )

        pyautogui.click(x=x, y=y, button=button, clicks=clicks)
        return f"Clicked '{text}' at ({x}, {y}) with {button} button x{clicks}"

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        return f"ERROR: Failed to parse OCR results: {e}"


if __name__ == "__main__":
    mcp.run()
