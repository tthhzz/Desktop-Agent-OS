"""LangGraphAgent — multi-agent implementation using LangGraph.

This agent replaces ``BasicMemoryAgent`` with a Supervisor + Worker
architecture powered by ``langgraph.StateGraph``.

Key features:
- **Supervisor** routes user requests to the appropriate worker.
- **Chat Worker** handles general conversation.
- **Tool Worker** executes MCP tools with optional Human-in-the-Loop approval.
- **LangSmith** tracing out of the box (set env vars).
- Fully compatible with the existing ``AgentInterface`` contract.
"""

import asyncio
from typing import AsyncIterator, Union, List, Dict, Any, Literal, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from ..agent_interface import AgentInterface
from ...output_types import SentenceOutput, DisplayText
from ...input_types import BatchInput, TextSource
from ...stateless_llm.stateless_llm_interface import StatelessLLMInterface
from ...stateless_llm.openai_compatible_llm import AsyncLLM as OpenAICompatibleAsyncLLM
from ....mcpp.tool_manager import ToolManager
from ....mcpp.tool_executor import ToolExecutor
from ...transformers import (
    sentence_divider,
    actions_extractor,
    tts_filter,
    display_processor,
)
from ....config_manager import TTSPreprocessorConfig
from ....chat_history_manager import get_history
from ....memory.memory_system import MemorySystem

from .graph import build_graph
from .state import AgentState
from .config import LangGraphAgentConfig


class LangGraphAgent(AgentInterface):
    """LangGraph-based multi-agent with Supervisor + Worker routing."""

    _system: str = "You are a helpful assistant."

    def __init__(
        self,
        llm: StatelessLLMInterface,
        system: str,
        live2d_model,
        tts_preprocessor_config: TTSPreprocessorConfig = None,
        faster_first_response: bool = True,
        segment_method: str = "pysbd",
        use_mcpp: bool = False,
        interrupt_method: Literal["system", "user"] = "user",
        tool_prompts: Dict[str, str] = None,
        tool_manager: Optional[ToolManager] = None,
        tool_executor: Optional[ToolExecutor] = None,
        mcp_prompt_string: str = "",
        human_in_the_loop: bool = True,
        high_risk_tools: List[str] = None,
        max_tool_rounds: int = 10,
        memory_system: Optional[MemorySystem] = None,
    ):
        super().__init__()
        self._memory: List[Dict[str, Any]] = []
        self._live2d_model = live2d_model
        self._tts_preprocessor_config = tts_preprocessor_config
        self._faster_first_response = faster_first_response
        self._segment_method = segment_method
        self._use_mcpp = use_mcpp
        self.interrupt_method = interrupt_method
        self._tool_prompts = tool_prompts or {}
        self._interrupt_handled = False

        self._tool_manager = tool_manager
        self._tool_executor = tool_executor
        self._mcp_prompt_string = mcp_prompt_string
        self._human_in_the_loop = human_in_the_loop
        self._high_risk_tools = high_risk_tools or []
        self._max_tool_rounds = max_tool_rounds
        self._memory_system = memory_system

        # Build LangChain-compatible LLM from our existing StatelessLLM config
        self._lc_llm = self._build_lc_llm(llm)

        self.set_system(system if system else self._system)
        self._build_graph()

        logger.info("LangGraphAgent initialized.")

    def _build_lc_llm(self, llm: StatelessLLMInterface) -> "ChatOpenAI":
        """Create a LangChain ChatOpenAI from our existing LLM configuration.

        We reuse the same API key, base_url, and model name so that both
        the LangGraph nodes and the streaming pipeline use the same backend.
        """
        if isinstance(llm, OpenAICompatibleAsyncLLM):
            lc_llm = ChatOpenAI(
                model=llm.model,
                base_url=llm.base_url,
                api_key=llm.client.api_key,
                temperature=llm.temperature,
                streaming=True,
            )
            logger.info(f"LangChain LLM created: model={llm.model}, base_url={llm.base_url}")
            return lc_llm
        else:
            # Fallback: try to read attributes
            model = getattr(llm, "model", "gpt-3.5-turbo")
            base_url = getattr(llm, "base_url", "https://api.openai.com/v1")
            api_key = getattr(llm, "client", None)
            if api_key and hasattr(api_key, "api_key"):
                api_key = api_key.api_key
            else:
                api_key = "sk-placeholder"

            lc_llm = ChatOpenAI(
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=getattr(llm, "temperature", 1.0),
                streaming=True,
            )
            logger.warning(f"LangChain LLM created with fallback config: model={model}")
            return lc_llm

    def _build_graph(self):
        """Build the LangGraph graph with current configuration."""
        # Prepare tools for the LLM (for bind_tools)
        self._lc_tools = []
        if self._use_mcpp and self._tool_manager:
            openai_tools = self._tool_manager.get_formatted_tools("OpenAI")
            self._lc_tools = openai_tools
            logger.info(f"LangGraph graph will use {len(self._lc_tools)} MCP tools")

        self._graph = build_graph(
            llm=self._lc_llm,
            system_prompt=self._system,
            tool_executor=self._tool_executor,
            tool_manager=self._tool_manager,
            high_risk_tools=self._high_risk_tools,
            human_in_the_loop=self._human_in_the_loop,
            memory_system=self._memory_system,
        )

    def set_system(self, system: str):
        """Set the system prompt."""
        if self.interrupt_method == "user":
            system = f"{system}\n\nIf you received `[interrupted by user]` signal, you were interrupted."
        self._system = system

    # ── Memory management (same interface as BasicMemoryAgent) ──

    def _add_message(self, message: Union[str, List[Dict[str, Any]]], role: str,
                     display_text: "DisplayText | None" = None, skip_memory: bool = False):
        if skip_memory:
            return
        text_content = ""
        if isinstance(message, list):
            for item in message:
                if item.get("type") == "text":
                    text_content += item["text"] + " "
            text_content = text_content.strip()
        elif isinstance(message, str):
            text_content = message
        else:
            return

        if not text_content and role == "assistant":
            return

        msg = {"role": role, "content": text_content}
        if display_text:
            if display_text.name:
                msg["name"] = display_text.name
            if display_text.avatar:
                msg["avatar"] = display_text.avatar

        if (
            self._memory
            and self._memory[-1]["role"] == role
            and self._memory[-1]["content"] == text_content
        ):
            return

        self._memory.append(msg)

    def set_memory_from_history(self, conf_uid: str, history_uid: str) -> None:
        messages = get_history(conf_uid, history_uid)
        self._memory = []
        for msg in messages:
            role = "user" if msg["role"] == "human" else "assistant"
            content = msg["content"]
            if isinstance(content, str) and content:
                self._memory.append({"role": role, "content": content})
        logger.info(f"Loaded {len(self._memory)} messages from history.")

    def handle_interrupt(self, heard_response: str) -> None:
        if self._interrupt_handled:
            return
        self._interrupt_handled = True
        if self._memory and self._memory[-1]["role"] == "assistant":
            self._memory[-1]["content"] = heard_response + "..."
        elif heard_response:
            self._memory.append({"role": "assistant", "content": heard_response + "..."})
        interrupt_role = "system" if self.interrupt_method == "system" else "user"
        self._memory.append({"role": interrupt_role, "content": "[Interrupted by user]"})

    def reset_interrupt(self) -> None:
        self._interrupt_handled = False

    # ── Input conversion ──────────────────────────────────────

    def _to_text_prompt(self, input_data: BatchInput) -> str:
        parts = []
        for td in input_data.texts:
            if td.source == TextSource.INPUT:
                parts.append(td.content)
            elif td.source == TextSource.CLIPBOARD:
                parts.append(f"[User shared content from clipboard: {td.content}]")
        if input_data.images:
            parts.append("\n[User has also provided images]")
        return "\n".join(parts).strip()

    def _input_to_state(self, input_data: BatchInput) -> AgentState:
        """Convert BatchInput into initial AgentState for the graph."""
        text_prompt = self._to_text_prompt(input_data)
        lc_messages = []

        # Load existing memory as LangChain messages
        for msg in self._memory:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            elif role == "system":
                lc_messages.append(SystemMessage(content=content))

        # Add current user input
        if text_prompt:
            lc_messages.append(HumanMessage(content=text_prompt))

            skip_memory = input_data.metadata and input_data.metadata.get("skip_memory", False)
            if not skip_memory:
                self._add_message(text_prompt, "user")

        images_data = None
        if input_data.images:
            images_data = [
                {"source": img.source.value, "data": img.data, "mime_type": img.mime_type}
                for img in input_data.images
            ]

        return {
            "messages": lc_messages,
            "next_worker": "",
            "tool_calls": [],
            "tool_results": [],
            "images": images_data,
            "metadata": input_data.metadata,
            "pending_approval": None,
            "approval_response": None,
            "retrieved_memories": None,
            "skill_match": None,
        }

    # ── Main chat method (with decorator pipeline) ────────────

    def _chat_function_factory(self):
        """Create the chat pipeline with the same decorator chain as BasicMemoryAgent."""

        @tts_filter(self._tts_preprocessor_config)
        @display_processor()
        @actions_extractor(self._live2d_model)
        @sentence_divider(
            faster_first_response=self._faster_first_response,
            segment_method=self._segment_method,
            valid_tags=["think"],
        )
        async def chat_with_graph(
            input_data: BatchInput,
        ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
            """Run the LangGraph and stream text output."""
            self.reset_interrupt()
            initial_state = self._input_to_state(input_data)
            logger.info(f"[LangGraph] ▶ input messages: {len(initial_state['messages'])}")

            # Run the graph and collect the final AI message
            try:
                result = await self._graph.ainvoke(initial_state)
            except Exception as e:
                logger.error(f"[LangGraph] ✗ execution error: {e}")
                yield f"[Error: {e}]"
                return

            # Extract the final AI message from the result
            final_messages = result.get("messages", [])
            next_worker = result.get("next_worker", "?")
            logger.info(f"[LangGraph] ◀ done | final msgs: {len(final_messages)} | last_worker: {next_worker}")

            ai_text = ""

            for msg in reversed(final_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    ai_text = msg.content
                    break

            if not ai_text:
                logger.warning("[LangGraph] ⚠ no AI message in result")
                return

            # Check for pending approval (Human-in-the-Loop)
            pending = result.get("pending_approval")
            if pending:
                # Yield the approval request as a special status
                yield {
                    "type": "tool_call_status",
                    "tool_id": "approval_request",
                    "tool_name": "approval_request",
                    "status": "pending_approval",
                    "content": pending.get("description", "Tool execution awaiting approval"),
                    "pending_approval": pending,
                }
                # Don't add to memory yet — conversation is paused
                return

            # Yield the text through the decorator pipeline
            self._add_message(ai_text, "assistant")
            yield ai_text

            # Update memory system after yielding (non-blocking)
            if self._memory_system:
                try:
                    # Extract user input
                    user_text = ""
                    for td in input_data.texts:
                        if td.source == TextSource.INPUT:
                            user_text += td.content + " "
                    user_text = user_text.strip()

                    # Extract tools used from result state
                    tools_used = []
                    for msg in reversed(final_messages):
                        if isinstance(msg, ToolMessage):
                            tools_used.append(msg.name if hasattr(msg, "name") else "tool")
                        elif isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tools_used.append(tc.get("name", ""))

                    await self._memory_system.on_conversation_turn(
                        user_input=user_text,
                        ai_response=ai_text,
                        tools_used=tools_used if tools_used else None,
                    )
                    logger.info(f"[Memory] on_conversation_turn saved (tools: {tools_used})")
                except Exception as e:
                    logger.error(f"[Memory] on_conversation_turn error: {e}")

        return chat_with_graph

    async def chat(
        self,
        input_data: BatchInput,
    ) -> AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]:
        """Run chat pipeline (implements AgentInterface.chat)."""
        chat_func = self._chat_function_factory()
        async for output in chat_func(input_data):
            yield output
