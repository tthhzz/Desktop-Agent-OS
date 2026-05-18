from typing import Literal
from loguru import logger

from .agents.agent_interface import AgentInterface
from .agents.langgraph_agent import LangGraphAgent
from .stateless_llm_factory import LLMFactory as StatelessLLMFactory

from ..mcpp.tool_manager import ToolManager
from ..mcpp.tool_executor import ToolExecutor
from ..memory.memory_system import MemorySystem
from typing import Optional


class AgentFactory:
    @staticmethod
    def create_agent(
        conversation_agent_choice: str,
        agent_settings: dict,
        llm_configs: dict,
        system_prompt: str,
        live2d_model=None,
        tts_preprocessor_config=None,
        **kwargs,
    ) -> AgentInterface:
        """Create an agent based on the configuration."""
        logger.info(f"Initializing agent: {conversation_agent_choice}")

        if conversation_agent_choice == "langgraph_agent":
            lg_settings: dict = agent_settings.get("langgraph_agent", {})
            llm_provider: str = lg_settings.get("llm_provider")

            if not llm_provider:
                raise ValueError("LLM provider not specified for langgraph agent")

            llm_config: dict = llm_configs.get(llm_provider)
            interrupt_method: Literal["system", "user"] = llm_config.pop(
                "interrupt_method", "user"
            )

            if not llm_config:
                raise ValueError(
                    f"Configuration not found for LLM provider: {llm_provider}"
                )

            llm = StatelessLLMFactory.create_llm(
                llm_provider=llm_provider, system_prompt=system_prompt, **llm_config
            )

            tool_prompts = kwargs.get("system_config", {}).get("tool_prompts", {})
            tool_manager: Optional[ToolManager] = kwargs.get("tool_manager")
            tool_executor: Optional[ToolExecutor] = kwargs.get("tool_executor")
            mcp_prompt_string: str = kwargs.get("mcp_prompt_string", "")
            memory_system: Optional[MemorySystem] = kwargs.get("memory_system")

            return LangGraphAgent(
                llm=llm,
                system=system_prompt,
                live2d_model=live2d_model,
                tts_preprocessor_config=tts_preprocessor_config,
                faster_first_response=lg_settings.get("faster_first_response", True),
                segment_method=lg_settings.get("segment_method", "pysbd"),
                use_mcpp=lg_settings.get("use_mcpp", False),
                interrupt_method=interrupt_method,
                tool_prompts=tool_prompts,
                tool_manager=tool_manager,
                tool_executor=tool_executor,
                mcp_prompt_string=mcp_prompt_string,
                human_in_the_loop=lg_settings.get("human_in_the_loop", True),
                high_risk_tools=lg_settings.get("high_risk_tools", []),
                max_tool_rounds=lg_settings.get("max_tool_rounds", 10),
                memory_system=memory_system,
            )

        else:
            raise ValueError(f"Unsupported agent type: {conversation_agent_choice}")
