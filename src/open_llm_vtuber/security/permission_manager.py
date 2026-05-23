"""Permission Manager — 4-level permission system for tool access control.

Levels:
- READ_ONLY: Search, read files, screenshot, screen info
- STANDARD: Memory read/write, terminal ls/cat, time
- SENSITIVE: File write, install packages, memory write
- DANGEROUS: Computer click/type/hotkey, terminal rm/shutdown, playwright navigate
"""

from enum import IntEnum
from typing import Dict, List, Optional, Set
from loguru import logger


class PermissionLevel(IntEnum):
    """Permission levels from most restrictive to least."""
    READ_ONLY = 0
    STANDARD = 1
    SENSITIVE = 2
    DANGEROUS = 3


# Default tool → permission level mapping
TOOL_PERMISSIONS: Dict[str, PermissionLevel] = {
    # READ_ONLY
    "time__get_current_time": PermissionLevel.READ_ONLY,
    "computer__screen_info": PermissionLevel.READ_ONLY,
    "computer__screenshot": PermissionLevel.READ_ONLY,
    "terminal__shell_read_file": PermissionLevel.READ_ONLY,
    "terminal__shell_list_dir": PermissionLevel.READ_ONLY,
    "terminal__shell_cwd": PermissionLevel.READ_ONLY,
    "ddg-search__search": PermissionLevel.READ_ONLY,
    "memory__memory_search": PermissionLevel.READ_ONLY,
    "memory__skill_list": PermissionLevel.READ_ONLY,
    "memory__skill_find": PermissionLevel.READ_ONLY,

    # STANDARD
    "memory__memory_write": PermissionLevel.STANDARD,

    # SENSITIVE
    "terminal__shell_write_file": PermissionLevel.SENSITIVE,
    "terminal__shell_exec": PermissionLevel.SENSITIVE,
    "playwright__browser_navigate": PermissionLevel.SENSITIVE,
    "playwright__browser_click": PermissionLevel.SENSITIVE,

    # DANGEROUS
    "computer__click": PermissionLevel.DANGEROUS,
    "computer__type": PermissionLevel.DANGEROUS,
    "computer__hotkey": PermissionLevel.DANGEROUS,
    "computer__scroll": PermissionLevel.DANGEROUS,
    "computer__move": PermissionLevel.DANGEROUS,
    "computer__drag": PermissionLevel.DANGEROUS,
}

# Per-session permission override (can be set via API)
# session_id → max allowed permission level
_session_overrides: Dict[str, PermissionLevel] = {}


class PermissionManager:
    """Manages tool access permissions with 4-level grading."""

    def __init__(
        self,
        tool_permissions: Optional[Dict[str, PermissionLevel]] = None,
        default_level: PermissionLevel = PermissionLevel.SENSITIVE,
    ):
        self._permissions = {**TOOL_PERMISSIONS, **(tool_permissions or {})}
        self._default_level = default_level
        self._audit_callback = None

    def set_audit_callback(self, callback):
        """Set a callback for audit logging of permission checks."""
        self._audit_callback = callback

    def get_level(self, tool_name: str) -> PermissionLevel:
        """Get the permission level required for a tool."""
        return self._permissions.get(tool_name, self._default_level)

    def check_permission(
        self,
        tool_name: str,
        session_level: PermissionLevel = PermissionLevel.SENSITIVE,
        session_id: str = "",
    ) -> bool:
        """Check if a tool call is allowed at the given session level.

        Returns True if the tool's required level <= session's allowed level.
        """
        # Check session override
        if session_id and session_id in _session_overrides:
            session_level = _session_overrides[session_id]

        required = self.get_level(tool_name)
        allowed = session_level >= required

        if self._audit_callback:
            self._audit_callback(
                tool_name=tool_name,
                required_level=required.name,
                session_level=session_level.name,
                allowed=allowed,
            )

        if not allowed:
            logger.warning(
                f"[Security] ✗ {tool_name} blocked "
                f"(requires {required.name}, session has {session_level.name})"
            )

        return allowed

    def classify_tools(self, tool_names: List[str]) -> Dict[str, List[str]]:
        """Classify a list of tool names by their permission level."""
        result = {level.name: [] for level in PermissionLevel}
        for name in tool_names:
            level = self.get_level(name)
            result[level.name].append(name)
        return result

    @staticmethod
    def set_session_level(session_id: str, level: PermissionLevel):
        """Override the permission level for a specific session."""
        _session_overrides[session_id] = level
        logger.info(f"[Security] session {session_id[:8]}... level set to {level.name}")

    @staticmethod
    def clear_session_level(session_id: str):
        """Remove a session-level override."""
        _session_overrides.pop(session_id, None)
