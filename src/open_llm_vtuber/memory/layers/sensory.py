"""Sensory Layer — short-lived perceptual buffer for camera/screen frames.

Stores the most recent N frames in a ring buffer.
Not persisted — cleared when the session ends.
"""

from collections import deque
from typing import Any, Dict, List, Optional
from loguru import logger


class SensoryLayer:
    """Ring buffer for recent sensory inputs (camera/screen frames)."""

    def __init__(self, buffer_size: int = 10):
        self._buffer: deque = deque(maxlen=buffer_size)
        self._buffer_size = buffer_size

    def add_frame(self, frame_data: str, source: str = "camera",
                  mime_type: str = "image/jpeg") -> None:
        """Add a frame to the sensory buffer."""
        self._buffer.append({
            "source": source,
            "data": frame_data,
            "mime_type": mime_type,
        })
        logger.debug(f"Sensory: added {source} frame (buffer: {len(self._buffer)}/{self._buffer_size})")

    def get_recent_frames(self, count: int = 1) -> List[Dict[str, Any]]:
        """Get the N most recent frames (newest first)."""
        count = min(count, len(self._buffer))
        frames = list(self._buffer)
        return list(reversed(frames))[:count]

    def clear(self) -> None:
        """Clear the sensory buffer."""
        self._buffer.clear()

    @property
    def frame_count(self) -> int:
        return len(self._buffer)
