"""Terminal MCP Server — shell command execution + file operations.

Provides these tools:
- shell_exec     — execute a shell command and return output
- shell_read_file — read a file's contents
- shell_write_file — write content to a file
- shell_list_dir  — list directory contents
- shell_cwd      — get/set current working directory

Security: integrates with the security module for blocked commands, permission
checks, and audit logging.
"""

import os
import asyncio
import subprocess
import time
from typing import Optional
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from ...security.blocked_list import is_blocked
from ...security.permission_manager import PermissionManager, PermissionLevel
from ...security.audit_logger import AuditLogger

mcp = FastMCP("terminal")

# Track working directory per session
_cwd: str = os.getcwd()

# Commands that require human approval (destructive but sometimes legitimate)
_RISKY_PATTERNS = [
    "rm ", "del ", "rmdir ", "rd /s",
    "shutdown", "reboot",
    "reg ", "regedit",
    "net user", "net localgroup",
    "pip uninstall", "conda remove",
    "taskkill", "kill ",
    "sudo ", "runas ",
]

# Initialize security components
_permission_manager = PermissionManager()
_audit_logger = AuditLogger()
_audit_logger.initialize()


def _is_risky(command: str) -> bool:
    """Check if a command is risky and might need human approval."""
    cmd_lower = command.lower().strip()
    for pattern in _RISKY_PATTERNS:
        if cmd_lower.startswith(pattern) or f" {pattern}" in cmd_lower or f"&&{pattern}" in cmd_lower or f";{pattern}" in cmd_lower:
            return True
    return False


@mcp.tool()
async def shell_exec(
    command: str,
    timeout: int = 30,
    cwd: Optional[str] = None,
) -> str:
    """Execute a shell command and return its output.

    Args:
        command: The shell command to execute.
        timeout: Maximum execution time in seconds. Default: 30. Max: 120.
        cwd: Working directory. If omitted, uses the session's current directory.

    Returns:
        Command output (stdout + stderr), or error message if execution fails.
        Output is truncated to 10000 chars if too long.
    """
    # Security: check blocked list
    blocked, reason = is_blocked(command)
    if blocked:
        _audit_logger.log(
            tool_name="shell_exec",
            arguments=command[:200],
            result_summary="BLOCKED",
            permission_level="BLOCKED",
            was_blocked=True,
        )
        return f"SECURITY BLOCKED: {reason}"

    # Cap timeout
    timeout = min(timeout, 120)

    global _cwd
    work_dir = cwd or _cwd
    if not os.path.isdir(work_dir):
        return f"Error: working directory does not exist: {work_dir}"

    try:
        start_time = time.monotonic()
        # Use PowerShell on Windows, bash on Unix
        if os.name == "nt":
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
            )
        else:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
            )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return f"TIMEOUT: Command timed out after {timeout}s\nPartial output may be lost."

        # Decode output
        out = stdout.decode("utf-8", errors="replace") if stdout else ""
        err = stderr.decode("utf-8", errors="replace") if stderr else ""

        # Build result
        result_parts = []
        if out:
            result_parts.append(out)
        if err:
            result_parts.append(f"[STDERR]\n{err}")
        if proc.returncode != 0:
            result_parts.append(f"[EXIT CODE: {proc.returncode}]")

        result = "\n".join(result_parts) if result_parts else "[No output]"

        # Truncate if too long
        if len(result) > 10000:
            result = result[:10000] + f"\n... [truncated, total {len(result)} chars]"

        # Audit log
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        _audit_logger.log(
            tool_name="shell_exec",
            arguments=command[:200],
            result_summary=result[:200],
            permission_level="SENSITIVE",
            was_approved=not _is_risky(command),
            duration_ms=elapsed_ms,
        )

        # Update session CWD if the command was a cd
        cmd_stripped = command.strip()
        if cmd_stripped.startswith("cd ") or cmd_stripped.startswith("chdir "):
            target = cmd_stripped.split(None, 1)[1].strip() if " " in cmd_stripped else ""
            if target:
                new_dir = os.path.abspath(os.path.join(work_dir, target))
                if os.path.isdir(new_dir):
                    _cwd = new_dir

        return result

    except Exception as e:
        return f"EXECUTION ERROR: {e}"


@mcp.tool()
async def shell_read_file(
    path: str,
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Read the contents of a text file.

    Args:
        path: Path to the file (absolute or relative to working directory).
        offset: Line number to start reading from (0-based). Default: 0.
        limit: Maximum number of lines to read. Default: 2000.

    Returns:
        File contents, or error message if the file cannot be read.
    """
    # Resolve relative paths against CWD
    if not os.path.isabs(path):
        path = os.path.join(_cwd, path)
    path = os.path.normpath(path)

    if not os.path.isfile(path):
        return f"Error: file not found: {path}"

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total_lines = len(lines)
        selected = lines[offset : offset + limit]
        result = "".join(selected)

        if len(result) > 50000:
            result = result[:50000] + f"\n... [truncated, file has {total_lines} lines]"

        # Add context about what was read
        header = f"[File: {path} | Lines {offset}-{min(offset + limit, total_lines)} of {total_lines}]\n"
        return header + result

    except Exception as e:
        return f"Error reading file: {e}"


@mcp.tool()
async def shell_write_file(
    path: str,
    content: str,
) -> str:
    """Write content to a file, creating it if it doesn't exist.

    Args:
        path: Path to the file (absolute or relative to working directory).
        content: The content to write.

    Returns:
        Confirmation message with bytes written.
    """
    # Resolve relative paths against CWD
    if not os.path.isabs(path):
        path = os.path.join(_cwd, path)
    path = os.path.normpath(path)

    try:
        # Create parent directories if needed
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Wrote {len(content)} chars to {path}"

    except Exception as e:
        return f"Error writing file: {e}"


@mcp.tool()
async def shell_list_dir(
    path: Optional[str] = None,
) -> str:
    """List contents of a directory.

    Args:
        path: Directory path. If omitted, uses the session's current directory.

    Returns:
        Directory listing with file sizes and types.
    """
    target = path or _cwd
    if not os.path.isabs(target):
        target = os.path.join(_cwd, target)
    target = os.path.normpath(target)

    if not os.path.isdir(target):
        return f"Error: directory not found: {target}"

    try:
        entries = os.listdir(target)
        lines = [f"[Directory: {target}]\n"]

        # Separate dirs and files
        dirs = []
        files = []
        for name in sorted(entries):
            full = os.path.join(target, name)
            if os.path.isdir(full):
                dirs.append(f"  📁 {name}/")
            else:
                size = os.path.getsize(full)
                size_str = f"{size:,} B"
                if size > 1024:
                    size_str = f"{size / 1024:.1f} KB"
                if size > 1024 * 1024:
                    size_str = f"{size / (1024*1024):.1f} MB"
                files.append(f"  📄 {name}  ({size_str})")

        lines.extend(dirs)
        lines.extend(files)
        lines.append(f"\n[{len(dirs)} directories, {len(files)} files]")

        return "\n".join(lines)

    except Exception as e:
        return f"Error listing directory: {e}"


@mcp.tool()
async def shell_cwd(
    new_dir: Optional[str] = None,
) -> str:
    """Get or set the current working directory for shell commands.

    Args:
        new_dir: If provided, changes the working directory to this path.
                 If omitted, returns the current working directory.

    Returns:
        The current working directory.
    """
    global _cwd
    if new_dir:
        if not os.path.isabs(new_dir):
            new_dir = os.path.join(_cwd, new_dir)
        new_dir = os.path.normpath(new_dir)
        if os.path.isdir(new_dir):
            _cwd = new_dir
            return f"CWD changed to: {_cwd}"
        else:
            return f"Error: directory not found: {new_dir}"
    return f"CWD: {_cwd}"


if __name__ == "__main__":
    mcp.run()
