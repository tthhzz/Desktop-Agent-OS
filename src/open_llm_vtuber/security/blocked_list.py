"""Blocked command list — hard-blocked patterns that can never be executed.

These are non-negotiable safety blocks. Even with DANGEROUS permission level,
these commands will never be allowed.
"""

from typing import List, Tuple

# Hard-blocked command patterns (case-insensitive substring match)
BLOCKED_COMMANDS: List[str] = [
    # System destruction
    "rm -rf /",
    "rm -rf /*",
    "del /s /q C:",
    "del /s /q D:",
    "format ",
    "mkfs.",
    ":(){ :|:& };:",  # fork bomb

    # Registry corruption
    "reg delete HKLM",
    "reg delete HKCU",

    # User/system manipulation
    "net user ",
    "net localgroup administrators",

    # Boot/shutdown
    "shutdown ",
    "reboot ",

    # Disk wipe
    "dd if=/dev/zero",
    "dd if=/dev/urandom",

    # Credential theft
    "cat /etc/shadow",
    "cat /etc/passwd",

    # Malware vectors
    "curl ",  # block curl to prevent remote script execution
    "wget ",
    "powershell -enc",
    "powershell -e ",
    "Invoke-WebRequest",
    "iex ",
    "Invoke-Expression",
]


def is_blocked(command: str) -> Tuple[bool, str]:
    """Check if a command contains a blocked pattern.

    Returns:
        (blocked, reason) — blocked=True if the command should be rejected.
    """
    cmd_lower = command.lower().strip()

    for pattern in BLOCKED_COMMANDS:
        if pattern.lower() in cmd_lower:
            return True, f"Blocked pattern: '{pattern}'"

    return False, ""
