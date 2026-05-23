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


@mcp.tool()
async def skill_validate(skill_id: str) -> str:
    """Validate a skill's structure and parameter definitions.

    Checks: required fields, param types, template placeholders.

    Args:
        skill_id: The skill ID to validate.

    Returns:
        Validation result: "Valid" or error description.
    """
    if not _memory_system or not _memory_system.skill:
        return "Error: Memory system not initialized"

    is_valid, reason = await _memory_system.skill.validate_skill(skill_id)
    return f"Skill validation: {'VALID' if is_valid else 'INVALID'} — {reason}"


@mcp.tool()
async def skill_evolve() -> str:
    """Run the skill self-evolution cycle: merge similar skills, prune unused, validate new ones.

    This is an advanced operation that optimizes the skill database.

    Returns:
        JSON summary of evolution actions: {merged, pruned, validated}.
    """
    if not _memory_system or not _memory_system.skill_miner:
        return "Error: Memory system not initialized"

    result = await _memory_system.skill_miner.evolve_skills()
    import json
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def skill_create(
    name: str,
    description: str,
    tools: str,
    trigger_condition: str,
    params: str = "[]",
    template: str = "",
) -> str:
    """Create a new parameterized skill.

    Args:
        name: Short snake_case name (e.g., "web_search_and_summarize").
        description: What this skill does.
        tools: Comma-separated tool names (e.g., "ddg-search,time").
        trigger_condition: When should this skill be activated.
        params: JSON array of parameter definitions. Default: "[]".
                Example: [{"name": "topic", "type": "string", "required": true}]
        template: Prompt template with {param} placeholders. Default: "".

    Returns:
        Confirmation with the skill ID.
    """
    if not _memory_system or not _memory_system.skill:
        return "Error: Memory system not initialized"

    import json

    tool_list = [t.strip() for t in tools.split(",") if t.strip()]

    try:
        param_list = json.loads(params) if params else []
    except json.JSONDecodeError:
        param_list = []

    record_id = await _memory_system.skill.register_skill(
        name=name,
        description=description,
        tools=tool_list,
        trigger_condition=trigger_condition,
        params=param_list,
        template=template or description,
    )
    return f"Created skill '{name}' (id={record_id[:8]}..., {len(tool_list)} tools, {len(param_list)} params)"


if __name__ == "__main__":
    mcp.run()
