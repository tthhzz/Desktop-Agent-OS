"""Memory MCP Server — exposes memory search/write/skill tools via MCP.

Tools:
- memory_search    — Search episodic and semantic memory
- memory_write     — Write a knowledge entry to semantic memory
- skill_list       — List all available skills
- skill_execute    — Look up a skill by description
"""

from typing import Optional
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("memory")

# Will be set during server initialization
_memory_system = None


def set_memory_system(memory_system):
    """Set the memory system instance (called by the server before running)."""
    global _memory_system
    _memory_system = memory_system


@mcp.tool()
async def memory_search(query: str, top_k: int = 5) -> str:
    """Search the AI's long-term memory for relevant information.

    Use this when you need to recall facts about the user,
    past conversations, or preferences.

    Args:
        query: What to search for (e.g., "user's favorite food", "weather discussions")
        top_k: Number of results to return. Default: 5.

    Returns:
        JSON string with search results from episodic and semantic memory.
    """
    if not _memory_system:
        return "Error: Memory system not initialized"

    context = await _memory_system.retrieve_context(query, top_k)

    import json
    results = {
        "episodic": context.get("episodic", []),
        "semantic": context.get("semantic", []),
        "matching_skills": context.get("skills", []),
    }
    return json.dumps(results, ensure_ascii=False, default=str)


@mcp.tool()
async def memory_write(content: str, categories: str = "", importance: float = 0.7) -> str:
    """Write a piece of knowledge to the AI's long-term semantic memory.

    Use this to remember important facts, preferences, or observations
    about the user.

    Args:
        content: The knowledge to store (e.g., "User prefers dark mode")
        categories: Comma-separated categories (e.g., "preference,ui"). Default: "".
        importance: Importance score 0.0-1.0. Default: 0.7.

    Returns:
        Confirmation with the stored knowledge ID.
    """
    if not _memory_system or not _memory_system.semantic:
        return "Error: Memory system not initialized"

    cats = [c.strip() for c in categories.split(",") if c.strip()] if categories else []
    record_id = await _memory_system.semantic.store_knowledge(
        content=content,
        source="explicit_write",
        categories=cats,
        importance=importance,
    )
    return f"Stored knowledge (id={record_id[:8]}...): {content}"


@mcp.tool()
async def skill_list() -> str:
    """List all available skills in the AI's skill memory.

    Returns:
        JSON array of all skills with names, descriptions, and trigger conditions.
    """
    if not _memory_system or not _memory_system.skill:
        return "Error: Memory system not initialized"

    skills = await _memory_system.skill.list_skills()
    import json
    return json.dumps(skills, ensure_ascii=False, default=str)


@mcp.tool()
async def skill_find(task_description: str) -> str:
    """Find a skill that matches a task description.

    Args:
        task_description: What the user wants to do (e.g., "check the weather")

    Returns:
        JSON array of matching skills.
    """
    if not _memory_system or not _memory_system.skill:
        return "Error: Memory system not initialized"

    skills = await _memory_system.skill.find_skill(task_description, top_k=3)
    import json
    return json.dumps(skills, ensure_ascii=False, default=str)


if __name__ == "__main__":
    mcp.run()
