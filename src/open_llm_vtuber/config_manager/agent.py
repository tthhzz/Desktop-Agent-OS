"""Agent configuration models."""

from pydantic import BaseModel, Field
from typing import Dict, ClassVar, Optional, Literal, List
from .i18n import I18nMixin, Description
from .stateless_llm import StatelessLLMConfigs


class LangGraphAgentConfig(I18nMixin, BaseModel):
    """Configuration for the LangGraph multi-agent."""

    llm_provider: Literal[
        "stateless_llm_with_template",
        "openai_compatible_llm",
        "claude_llm",
        "llama_cpp_llm",
        "ollama_llm",
        "lmstudio_llm",
        "openai_llm",
        "gemini_llm",
        "zhipu_llm",
        "deepseek_llm",
        "groq_llm",
        "mistral_llm",
    ] = Field(..., alias="llm_provider")

    faster_first_response: Optional[bool] = Field(True, alias="faster_first_response")
    segment_method: Literal["regex", "pysbd"] = Field("pysbd", alias="segment_method")
    use_mcpp: Optional[bool] = Field(False, alias="use_mcpp")
    mcp_enabled_servers: Optional[List[str]] = Field([], alias="mcp_enabled_servers")
    human_in_the_loop: Optional[bool] = Field(True, alias="human_in_the_loop")
    high_risk_tools: Optional[List[str]] = Field(
        default=["computer__click", "computer__type", "computer__hotkey", "playwright__navigate"],
        alias="high_risk_tools",
    )
    max_tool_rounds: Optional[int] = Field(10, alias="max_tool_rounds")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "llm_provider": Description(
            en="LLM provider to use for this agent",
            zh="LangGraph 智能体使用的大语言模型选项",
        ),
        "human_in_the_loop": Description(
            en="Whether to require user approval for high-risk tool calls (default: True)",
            zh="高风险工具调用是否需要用户审批（默认：True）",
        ),
        "high_risk_tools": Description(
            en="List of tool names that require human approval before execution",
            zh="需要用户审批才能执行的工具名称列表",
        ),
        "max_tool_rounds": Description(
            en="Maximum number of tool call rounds before forcing a final answer (default: 10)",
            zh="强制生成最终答案前的最大工具调用轮数（默认：10）",
        ),
    }


class AgentSettings(I18nMixin, BaseModel):
    """Settings for different types of agents."""

    langgraph_agent: Optional[LangGraphAgentConfig] = Field(
        None, alias="langgraph_agent"
    )

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "langgraph_agent": Description(
            en="Configuration for LangGraph multi-agent", zh="LangGraph 多智能体配置"
        ),
    }


class AgentConfig(I18nMixin, BaseModel):
    """This class contains all of the configurations related to agent."""

    conversation_agent_choice: Literal["langgraph_agent"] = Field(
        ..., alias="conversation_agent_choice"
    )
    agent_settings: AgentSettings = Field(..., alias="agent_settings")
    llm_configs: StatelessLLMConfigs = Field(..., alias="llm_configs")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "conversation_agent_choice": Description(
            en="Type of conversation agent to use", zh="要使用的对话代理类型"
        ),
        "agent_settings": Description(
            en="Settings for different agent types", zh="不同代理类型的设置"
        ),
        "llm_configs": Description(
            en="Pool of LLM provider configurations", zh="语言模型提供者配置池"
        ),
    }
