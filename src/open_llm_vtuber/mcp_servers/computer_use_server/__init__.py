"""Computer Use MCP Server — desktop automation via pyautogui.

Provides these tools:
- computer_screenshot   — capture screen or region
- computer_click        — click at (x, y)
- computer_type         — type text via keyboard
- computer_hotkey       — press key combination (e.g. "ctrl+c")
- computer_scroll       — scroll up/down
- computer_screen_info  — get screen resolution
"""

import base64
import io
import asyncio
from typing import Optional

import pyautogui
from PIL import Image
from mcp.server.fastmcp import FastMCP

# Safety: give pyautogui a failsafe (move mouse to corner to abort)
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1  # Small pause between actions for stability

mcp = FastMCP("computer-use")


@mcp.tool()
def computer_screenshot(region: Optional[str] = None) -> str:
    """Take a screenshot of the entire screen or a region.

    Args:
        region: Optional region as "left,top,width,height" (e.g. "0,0,800,600").
                If omitted, captures the full screen.

    Returns:
        Base64-encoded PNG image of the screenshot.
    """
    if region:
        parts = [int(x.strip()) for x in region.split(",")]
        if len(parts) != 4:
            return "Error: region must be 'left,top,width,height'"
        img = pyautogui.screenshot(region=tuple(parts))
    else:
        img = pyautogui.screenshot()

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


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
    pyautogui.click(x=x, y=y, button=button, clicks=clicks)
    return f"Clicked {button} at ({x}, {y}) x{clicks}"


@mcp.tool()
def computer_type(text: str, interval: float = 0.02) -> str:
    """Type text using the keyboard.

    Args:
        text: The text to type.
        interval: Seconds between keystrokes. Default: 0.02.

    Returns:
        Confirmation message.
    """
    pyautogui.typewrite(text, interval=interval)
    return f"Typed: {text[:50]}{'...' if len(text) > 50 else ''}"


@mcp.tool()
def computer_hotkey(*keys: str) -> str:
    """Press a keyboard shortcut / hotkey combination.

    Args:
        keys: Keys to press simultaneously, e.g. "ctrl" "c" or "alt" "tab".

    Returns:
        Confirmation message.
    """
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


if __name__ == "__main__":
    mcp.run()
