"""Configuration dataclass for LangGraphAgent."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class LangGraphAgentConfig:
    """Runtime configuration for the LangGraph-based multi-agent.

    This is populated from ``conf.yaml → agent_settings.langgraph_agent``.
    """

    # LLM provider key in llm_configs
    llm_provider: str = "openai_compatible_llm"

    # MCP
    use_mcpp: bool = False
    mcp_enabled_servers: List[str] = field(default_factory=list)

    # Text processing
    faster_first_response: bool = True
    segment_method: str = "pysbd"

    # Human-in-the-Loop
    human_in_the_loop: bool = True
    high_risk_tools: List[str] = field(default_factory=lambda: [
        "computer__click",
        "computer__type",
        "computer__hotkey",
        "playwright__navigate",
    ])

    # Interrupt
    interrupt_method: str = "user"  # "system" | "user"

    # Max tool-call rounds before forcing a final answer
    max_tool_rounds: int = 10
