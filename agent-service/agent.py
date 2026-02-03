"""
Agent core logic using LangGraph + MCP.

This module implements the "agent-service" component in the 3‑service
architecture:

- web-ui/app.py (Streamlit)   → HTTP client
- agent-service/main.py       → FastAPI wrapper around this module
- mcp-server/server.py        → MCP tools for data analysis

The DataAnalystAgent below connects to one or more MCP servers, discovers
their tools, and builds a LangGraph ReAct agent on top. The agent receives
natural‑language questions about data and decides which MCP tools to call.
"""

from __future__ import annotations

import asyncio
import functools
import base64
import copy
import json
import uuid
import re
#import httpx
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Annotated

import nest_asyncio
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import ToolException
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError


from config import load_config

# Apply nest_asyncio to allow running asyncio code in interactive shells
nest_asyncio.apply()

# Location where the agent-service caches artifacts fetched from the MCP server
ARTIFACT_CACHE_ROOT = Path("/tmp/agent_artifacts")
ARTIFACT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

MAX_VALIDATE_RETRIES = 3

# Create a state variable for the LangGraph Agent
class AgentState(TypedDict):
    """
    Mutable LangGraph state carried between nodes.

    Keys:
        messages: running transcript of System/Human/AI/Tool messages.
        retries: number of consecutive validation retries issued to the model.
        data_registered: whether `register_external_data` succeeded for the session.
    """

    messages: List[Any]
    retries: int
    data_registered: bool

# ============================================================================
# Configuration Helpers
# ============================================================================


def _load_runtime_config() -> Dict[str, Any]:
    """
    Retrieve the merged configuration for the agent service.

    Falls back to sensible defaults if keys are missing so the agent remains
    operational during incremental configuration rollouts.
    """
    config = load_config()
    llm_cfg = config.get("llm", {}) or {}
    mcp_cfg = config.get("mcp", {}) or {}

    provider_name = llm_cfg.get("provider", "openai")
    provider = (provider_name or "openai").lower()

    provider_section = llm_cfg.get(provider, {})
    if not isinstance(provider_section, dict):
        provider_section = {}

    legacy_api_base = llm_cfg.get("api_base")
    legacy_api_key = llm_cfg.get("api_key")
    legacy_aws_cfg = llm_cfg.get("aws", {})

    # Normalize endpoints into a list
    endpoints_raw = mcp_cfg.get("endpoints", ["http://127.0.0.1:4242/mcp"])
    if isinstance(endpoints_raw, str):
        endpoints = [item.strip() for item in endpoints_raw.split(",") if item.strip()]
    else:
        endpoints = list(endpoints_raw)

    # Shared LLM defaults
    model_name = llm_cfg.get("model", "gpt-4o")
    temperature_raw = llm_cfg.get("temperature", 0.7)
    try:
        temperature = float(temperature_raw)
    except (TypeError, ValueError):
        temperature = 0.7

    # Provider-specific settings
    api_base: Optional[str] = None
    api_key: Optional[str] = None

    if provider in {"openai", "vllm"}:
        api_base = provider_section.get("api_base") or legacy_api_base or "https://api.openai.com/v1"
        api_key = provider_section.get("api_key")
        if api_key is None:
            api_key = legacy_api_key
        api_key = api_key or None
    elif provider == "bedrock":
        api_base = provider_section.get("endpoint_url") or (
            legacy_aws_cfg.get("endpoint_url") if isinstance(legacy_aws_cfg, dict) else None
        )
        api_key = None
    else:
        api_base = provider_section.get("api_base") or legacy_api_base
        api_key = provider_section.get("api_key")
        if api_key is None:
            api_key = legacy_api_key
        api_key = api_key or None

    aws_defaults = {
        "region": None,
        "profile": None,
        "access_key_id": None,
        "secret_access_key": None,
        "session_token": None,
        "endpoint_url": None,
    }
    merged_aws: Dict[str, Any] = dict(aws_defaults)

    if isinstance(legacy_aws_cfg, dict):
        for key, value in legacy_aws_cfg.items():
            if key in merged_aws:
                merged_aws[key] = value

    if provider == "bedrock":
        for key, value in provider_section.items():
            if key in merged_aws:
                merged_aws[key] = value

    llm_settings = {
        "provider": provider_name,
        "api_base": api_base,
        "model": model_name,
        "temperature": temperature,
        "api_key": api_key,
        "aws": merged_aws,
    }

    retries_cfg = {
        "connect_timeout_seconds": int(mcp_cfg.get("connect_timeout_seconds", 30)),
        "max_retries": int(mcp_cfg.get("max_retries", 5)),
    }

    return {
        "llm": llm_settings,
        "mcp_endpoints": endpoints or ["http://127.0.0.1:4242/mcp"],
        "retries": retries_cfg,
    }

# ============================================================================
# Create Chat Model that can handle async tool calling with vllm hosted models
# ============================================================================

class ChatOpenAIVLLM(ChatOpenAI):
    """
    ChatOpenAI subclass that:
    - preserves bind_tools() and tool calling
    - overrides async execution to call vLLM synchronously
    """

    async def _agenerate(self, messages, **kwargs) -> ChatResult:
        """Execute the synchronous OpenAI-compatible call in a worker thread."""
        loop = asyncio.get_running_loop()

        # Run the SYNC OpenAI-compatible call in a thread
        result = await loop.run_in_executor(
            None,
            functools.partial(
                super()._generate,
                messages,
                **kwargs,
            ),
        )

        return result


def _build_chat_model(llm_config):
    """Instantiate the LLM client configured for the requested provider."""
    provider = (llm_config.get("provider") or "openai").lower()

    if provider == "bedrock":
        aws_cfg = llm_config.get("aws") or {}
        bedrock_kwargs: Dict[str, Any] = {}

        region = aws_cfg.get("region")
        if region:
            bedrock_kwargs["region_name"] = region

        access_key = aws_cfg.get("access_key_id")
        if access_key:
            bedrock_kwargs["aws_access_key_id"] = access_key

        secret_key = aws_cfg.get("secret_access_key")
        if secret_key:
            bedrock_kwargs["aws_secret_access_key"] = secret_key

        session_token = aws_cfg.get("session_token")
        if session_token:
            bedrock_kwargs["aws_session_token"] = session_token

        endpoint_url = aws_cfg.get("endpoint_url")
        if endpoint_url:
            bedrock_kwargs["endpoint_url"] = endpoint_url

        model_kwargs: Dict[str, Any] = {}
        temperature = llm_config.get("temperature")
        if temperature is not None:
            model_kwargs["temperature"] = temperature
        model_kwargs["max_tokens"] = 4096 # Set a higher max_tokens to prevent truncation of tool calls
        if model_kwargs:
            bedrock_kwargs["model_kwargs"] = model_kwargs

        return ChatBedrock(
            model_id=llm_config["model"],
            **bedrock_kwargs,
        )

    if provider in {"openai", "vllm"}:
        return ChatOpenAIVLLM(
            model=llm_config["model"],
            temperature=llm_config["temperature"],
            base_url=llm_config["api_base"],
            api_key=llm_config.get("api_key"),
            #http_client=httpx.Client(verify=False)
        )

    raise ValueError(f"Unsupported LLM provider '{llm_config.get('provider')}'.")

# ============================================================================
# DataAnalystAgent Class
# ============================================================================

class DataAnalystAgent:
    """
    Agent for analyzing data using MCP tools.

    This class wraps three main responsibilities:

    1. Connect to one or more MCP servers and discover their tools.
    2. Build a LangGraph ReAct agent over those tools and an LLM.
    3. Provide a simple `analyze(query, session_id)` API for the FastAPI layer.

    The agent does not store any data itself; it only orchestrates calls to
    MCP tools that operate on per‑session CSV files.
    """
    
    def __init__(self, mcp_urls: List[str], llm_config: Dict[str, Any], retry_config: Dict[str, Any]):
        """
        Initializes the DataAnalystAgent.

        Args:
            mcp_urls: A list of MCP server URLs.
            llm_config: Configuration for the underlying LLM.
            retry_config: Settings controlling connection retries.
        """
        self.clients: List[MultiServerMCPClient] = []
        self.tools = None
        self.agent = None
        self.llm_config = llm_config
        self.retry_config = retry_config
        self.client_by_server: Dict[str, MultiServerMCPClient] = {}
        self.tool_server_map: Dict[str, str] = {}
        self.artifact_registry: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # Underlying chat model used by the ReAct agent.
        self.llm = _build_chat_model(llm_config)
        self._initializing = False
        # List of MCP HTTP endpoints the agent can try in order.
        self.mcp_urls = mcp_urls
        self._tool_node: Optional[ToolNode] = None
        self.prompt_template: Optional[ChatPromptTemplate] = None


    async def _llm_node(
        self,
        state: AgentState,
        config: Optional[RunnableConfig] = None,
    ) -> AgentState:
        """Run one LLM turn and append the resulting `AIMessage` to the transcript."""
        messages = state.get("messages", [])
        if self._is_bedrock_backend():
            messages = [self._sanitize_tool_message_for_bedrock(m) if isinstance(m, ToolMessage) else m for m in messages]
        
        # If the last message was a tool message, add a human message to give context to the model
        if messages and isinstance(messages[-1], ToolMessage):
            tool_output = messages[-1].content
            messages.append(HumanMessage(f"Tool execution resulted in the following output:\n```\n{tool_output}\n```\nBased on this, what is the next step in your analysis? Cite the output in your reasoning."))

        ai_msg = await self.llm.ainvoke(messages)
        # Debug print to look at what the AI model is producing in its chats
        # print(f"\n {ai_msg} \n")
        return {
            "messages": messages + [ai_msg],
            "retries": state.get("retries", 0),
            "data_registered": state.get("data_registered", False),
        }


    async def _tool_feedback(self, state: AgentState) -> AgentState:
        """Add corrective feedback when the model attempted a tool call but failed."""
        feedback = (
            "The previous tool call could not be executed.\n\n"
            "If you intended to call a tool, you MUST emit exactly one tool call "
            "using the correct tool name and arguments.\n"
            "Do not explain the call. Do not narrate.\n"
            "Retry now."
        )

        messages = state.get("messages", [])
        return {
            "messages": messages + [HumanMessage(content=feedback)],
            "retries": state.get("retries", 0) + 1,
            "data_registered": state.get("data_registered", False),
        }

    
    def _extract_tool_call_from_text(self, message: AIMessage) -> Optional[dict]:
        """
        Attempts to recover a tool call from raw text when `tool_calls` are empty.
        This is necessary because some LLMs (e.g., Llama 4 on vLLM) may output
        tool calls as raw text instead of structured `tool_calls` fields.

        Supports:
        - strict JSON: `{"name":"...", "arguments": {...}}`
        - pythonic calls: `tool_name(arg1="x", arg2=3)`

        Returns normalized dict: `{"name": str, "arguments": dict}`
        """
        content = message.content
        text_fragments: List[str] = []

        if isinstance(content, str):
            text_fragments.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    text_fragments.append(item)
                elif isinstance(item, dict):
                    text_value = item.get("text") or item.get("content")
                    if isinstance(text_value, str):
                        text_fragments.append(text_value)
        else:
            return None

        text = "\n".join(fragment.strip() for fragment in text_fragments if isinstance(fragment, str)).strip()
        if not text:
            return None

        # 1) Try strict JSON first (preferred for modern LLMs)
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                name = data.get("name") or data.get("tool") or data.get("tool_name")
                args = data.get("arguments") or data.get("args") or data.get("parameters")
                if isinstance(name, str) and isinstance(args, dict):
                    return {"name": name, "arguments": args}
        except Exception:
            pass

        # 2) Try pythonic style parser: tool_name(arg="val", count=3)
        # This is crucial for models like Llama 4 on vLLM that primarily emit
        # tool calls in this format.
        # The regex is conservative to avoid false positives: it matches
        # an entire string that starts with a valid Python identifier,
        # followed by parentheses containing arguments.
        pythonic_re = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*(.*)\s*\)\s*$", re.DOTALL)
        m = pythonic_re.match(text)
        if not m:
            return None

        tool_name = m.group(1)
        args_str = m.group(2).strip()
        if not args_str:
            return {"name": tool_name, "arguments": {}}

        # Simple parser for key=value pairs within the pythonic call.
        # It handles nested structures by tracking depth of parentheses/brackets.
        args = {}
        parts = []
        depth = 0
        cur = []
        for ch in args_str:
            if ch == "," and depth == 0:
                parts.append("".join(cur).strip())
                cur = []
                continue
            cur.append(ch)
            if ch in ("(", "[", "{"):
                depth += 1
            elif ch in (")", "]", "}"):
                depth -= 1
        if cur:
            parts.append("".join(cur).strip())

        for part in parts:
            if "=" not in part:
                # Positional arguments are not reliably supported by this parser.
                # Only keyword arguments are processed.
                continue
            k, v = part.split("=", 1)
            k = k.strip()
            v = v.strip()
            # Strip quotes if present
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            else:
                # Try to interpret as JSON number / bool / list / dict
                try:
                    v_parsed = json.loads(v)
                    v = v_parsed
                except Exception:
                    pass
            args[k] = v

        return {"name": tool_name, "arguments": args}

    def _render_message_content(self, message: Optional[Any]) -> str:
        """
        Normalize the content emitted by LangChain messages into a plain string.

        LangChain message content can be a string, a list of strings/dicts, or other
        structured values (e.g., OpenAI content blocks). This helper centralizes the
        conversion so call sites do not have to repeat the extraction logic.
        """
        if message is None:
            return ""

        content = getattr(message, "content", "")
        if isinstance(content, list):
            fragments: List[str] = []
            for part in content:
                if isinstance(part, str):
                    fragments.append(part)
                elif isinstance(part, dict):
                    text_value = part.get("text")
                    if isinstance(text_value, str):
                        fragments.append(text_value)
                    else:
                        dict_content = part.get("content")
                        if isinstance(dict_content, str):
                            fragments.append(dict_content)
            return "\n".join(fragments).strip()
        if isinstance(content, str):
            return content.strip()
        return str(content)





    def _is_bedrock_backend(self) -> bool:
        return (self.llm_config.get("provider") or "").lower() == "bedrock"


    def _sanitize_tool_message_for_bedrock(self, message: ToolMessage) -> ToolMessage:
        """Return a copy of the tool message without LangChain-only metadata, which breaks the bedrock backend."""
        def _scrub(value):
            if isinstance(value, dict):
                return {k: _scrub(v) for k, v in value.items() if k != "id"}
            if isinstance(value, list):
                return [_scrub(item) for item in value]
            return value

        sanitized_content = _scrub(copy.deepcopy(getattr(message, "content", None)))
        sanitized_blocks = _scrub(copy.deepcopy(getattr(message, "content_blocks", None)))
        return ToolMessage(
            content=sanitized_content,
            tool_call_id=message.tool_call_id,
            name=message.name,
            id=message.id,
            artifact=getattr(message, "artifact", None),
            status=getattr(message, "status", "success"),
            additional_kwargs=copy.deepcopy(getattr(message, "additional_kwargs", {})),
            response_metadata=copy.deepcopy(getattr(message, "response_metadata", {})),
            content_blocks=sanitized_blocks,
        )

    
    async def _tools_node(
        self,
        state: AgentState,
        config: Optional[RunnableConfig] = None,
    ) -> AgentState:
        """Wrapper around LangGraph's ToolNode to reset retry counters after execution."""
        if self._tool_node is None:
            raise RuntimeError("Tool node not initialized.")

        try:
            result = await self._tool_node.ainvoke(state, config=config)
            new_messages = result.get("messages") or []
            combined_messages = list(state.get("messages", [])) + list(new_messages)

            # Track whether register_external_data has already succeeded so downstream
            # steps can suppress redundant registration attempts.
            data_registered_flag = state.get("data_registered", False)
            for message in new_messages:
                if isinstance(message, ToolMessage):
                    tool_name = message.name or ""
                    if tool_name.endswith("register_external_data"):
                        status = getattr(message, "status", "success")
                        if status != "error":
                            data_registered_flag = True

            updated_state: AgentState = {
                "messages": combined_messages,
                "retries": 0,
                "data_registered": data_registered_flag,
            }
            return updated_state
        except (ToolException, ValidationError, ValueError) as exc:
            messages = list(state.get("messages", []))
            tool_call_id = "unknown"

            if messages:
                last = messages[-1]
                if isinstance(last, AIMessage):
                    call_entries = list(last.tool_calls or [])
                    if call_entries:
                        entry = call_entries[0]
                        if isinstance(entry, dict):
                            tool_call_id = entry.get("id", "unknown")
                        else:
                            tool_call_id = getattr(entry, "id", "unknown") or "unknown"

            error_text = f"Tool execution failed: {type(exc).__name__}: {exc}"
            tool_error = ToolMessage(tool_call_id=tool_call_id, content=error_text)
            feedback = HumanMessage(
                content=(
                    "The previous tool call could not be executed due to the error above.\n"
                    "Return a single valid tool call with the required arguments."
                )
            )
            return {
                "messages": messages + [tool_error, feedback],
                "retries": state.get("retries", 0) + 1,
                "data_registered": state.get("data_registered", False),
            }


    async def _connect_to_mcp_servers(
        self,
        max_retries: Optional[int],
    ) -> List[Any]:
        """
        Contact each configured MCP endpoint and collect their tools.

        Returns:
            A flat list of LangChain tool instances compatible with LangGraph.

        Raises:
            RuntimeError: If all connections fail even after the configured retries.
        """
        connect_timeout = self.retry_config.get("connect_timeout_seconds", 30)
        retry_limit = max(1, max_retries or self.retry_config.get("max_retries", 5))

        combined_tools: List[Any] = []
        self.clients = []
        self.client_by_server = {}
        self.tool_server_map = {}

        last_error: Optional[Exception] = None

        for index, url in enumerate(self.mcp_urls, start=1):
            server_id = f"mcp_{index}"
            descriptor = {
                server_id: {
                    "transport": "streamable_http",
                    "url": url,
                }
            }
            connected = False
            for attempt in range(retry_limit):
                try:
                    attempt_label = f"(attempt {attempt + 1}/{retry_limit})" if retry_limit > 1 else ""
                    print(f"🔗 Connecting to MCP server at {url} {attempt_label}...")
                    client = MultiServerMCPClient(descriptor)
                    tools = await asyncio.wait_for(client.get_tools(), timeout=connect_timeout)
                    if not tools:
                        print(f"⚠️ MCP server {url} returned no tools.")
                        break
                    self.client_by_server[server_id] = client
                    self.clients.append(client)
                    combined_tools.extend(tools)
                    for tool in tools:
                        self.tool_server_map[tool.name] = server_id
                    print(f"✅ Loaded {len(tools)} tools from MCP server at {url}")
                    connected = True
                    break
                except asyncio.TimeoutError as exc:
                    last_error = exc
                    print(f"⏱️ Connection timeout for MCP server at {url}.")
                except Exception as exc:
                    last_error = exc
                    print(f"⚠️ Connection failed for MCP server at {url}: {type(exc).__name__} - {exc}")

                if attempt < retry_limit - 1:
                    await asyncio.sleep(2 ** attempt)

            if not connected:
                print(f"❌ Skipping MCP server at {url} after {retry_limit} failed attempts.")

        if not combined_tools:
            raise RuntimeError(
                f"Failed to connect to any MCP server. Last error: {last_error}"
            )

        return combined_tools

    def _build_system_prompt(self) -> str:
        """Compose the system prompt that defines the agent's behaviour."""
        return (
            "You are a concise and meticulous data analyst with access to multiple MCP tool servers. "
            "Your primary goal is to assist the user by following a cycle of thought, action (tool use), and observation to answer their request. Follow these rules:\n"
            "1. Always include the current `session_id` when invoking MCP tools so work stays scoped to the user.\n"
            "2. Inspect available data before acting—use `list_available_files` or `get_file_schema` when unsure about inputs.\n"
            "3. Load datasets with pandas based on their actual format (CSV, TSV, Excel, JSON, Parquet, Feather, etc.) rather than assuming CSV.\n"
            "4. When using `analyze_session_data`, pass the resolved local paths and write any outputs under `/tmp/mcp_outputs/<session>`, if there need to be any outputs.\n"
            "5. If you need access to any outputs like a file, plot, or dataset, publish it using `publish_artifact(session_id=..., path=...)`.\n"
            "6. Call `register_external_data` only when the user provides new external dataset URLs that are not yet cached.\n"
            "7. Prefer concise, step-by-step reasoning and cite the tool outputs that support your conclusions.\n"
            "8. When you need to show code or a tool call as part of your explanation and not execute it, you MUST wrap it in a markdown code block with the language specified. For example: \n```python\nprint('This is an example and will not be executed.')\n```"
        )

    def _build_react_graph(self) -> Any:
        """Construct and compile the LangGraph ReAct workflow."""
        graph = StateGraph(AgentState)

        graph.add_node("assistant", self._llm_node)
        graph.add_node("tools", self._tools_node)
        graph.add_node("tool_feedback", self._tool_feedback)

        graph.set_entry_point("assistant")

        graph.add_conditional_edges(
            "assistant",
            self._route_from_assistant,
            {
                "tools": "tools",
                "tool_feedback": "tool_feedback",
                END: END,
            },
        )

        graph.add_edge("tool_feedback", "assistant")
        graph.add_edge("tools", "assistant")

        return graph.compile(checkpointer=MemorySaver())

    
    def _route_from_assistant(self, state: AgentState):
        """Decide the next node after an assistant turn."""
        messages = state.get("messages", [])
        if not messages:
            return END

        last = messages[-1]
        if isinstance(last, AIMessage):
            if last.tool_calls:
                return "tools"

            if (
                state.get("retries", 0) < MAX_VALIDATE_RETRIES
                and self._extract_tool_call_from_text(last)
            ):
                return "tool_feedback"

        return END


    
    async def initialize(self, max_retries: Optional[int] = None):
        """
        Initializes the MCP client and agent with retry logic.

        Args:
            max_retries: The maximum number of times to retry connecting to the MCP server.
        """
        # Do nothing if initialization already happened or is in progress.
        if self.agent is not None or self._initializing:
            return
        self._initializing = True
        try:
            combined_tools = await self._connect_to_mcp_servers(max_retries)
            self.tools = combined_tools
            self.llm = self.llm.bind_tools(self.tools)
            self._tool_node = ToolNode(self.tools)

            system_prompt = self._build_system_prompt()
            self.prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            # The template is retained for potential future multi-turn usage; the analyze method
            # currently constructs a teaching prompt per request for clarity.

            self.agent = self._build_react_graph()
            print(f"Created ReAct agent with {len(self.tools)} tools from {len(self.clients)} MCP servers")
        finally:
            self._initializing = False
        return
    

    async def analyze(self, query: str, session_id: str) -> dict:
        """
        Run a full ReAct loop for the provided query and session.

        Args:
            query: Natural-language question or instruction from the user.
            session_id: Unique identifier for the conversation / workspace.

        Returns:
            Mapping with keys:
                text: natural-language answer string returned by the agent.
                plot: reserved for future structured visual output (currently always None).
                artifacts: list of published artifact descriptors for the session.
        """
        if self.agent is None:
            # Lazy initialization on first use so the FastAPI layer can start quickly.
            await self.initialize()

        # Retrieve the previous conversation state for this session.
        state = await self.agent.aget_state(config={"configurable": {"thread_id": session_id}})
        state_values: Dict[str, Any] = state.values if state and hasattr(state, "values") else {}
        existing_messages: List[Any] = list(state_values.get("messages", []))

        # Build session-specific guidance that we append to the base system prompt.
        # This supplements the stable system instructions with dynamic reminders
        # about the session identifier and cached datasets.
        session_guidance = (
            f"For this request, use session_id '{session_id}' in every MCP tool call so work remains scoped."
        )
        data_registered = bool(state_values.get("data_registered", False))
        if data_registered:
            session_guidance += (
                " External data for this session is already cached; reuse those resources and only call "
                "register_external_data if the user supplies new dataset URLs."
            )
        else:
            session_guidance += (
                " No external data has been cached yet. If the user provides dataset URLs, call "
                "register_external_data(session_id=..., assets=[...]) once to stage them before analysis."
            )

        base_system_prompt = self._build_system_prompt()
        system_message = SystemMessage(content=f"{base_system_prompt}\n\nSession guidance: {session_guidance}")

        # Strip prior system messages; we'll resend the up-to-date system guidance.
        history = [m for m in existing_messages if not isinstance(m, SystemMessage)]
        messages_to_send: List[Any] = [system_message] + history + [HumanMessage(content=query)]

        try:
            # Invoke the LangGraph ReAct agent. The `thread_id` ensures the graph can keep state per user session if needed.
            existing_ids = set(self.artifact_registry.get(session_id, {}).keys())
            response = await self.agent.ainvoke(
                {
                    "messages": messages_to_send,
                    "retries": 0,
                    "data_registered": data_registered,
                },
                config={
                    "recursion_limit": 100,
                    "configurable": {"thread_id": session_id},
                },
            )

            # Extract the final natural-language answer and (optionally) a plot.
            messages = response.get("messages", [])
            final_message = messages[-1] if messages else None
            artifacts = await self._capture_and_cache_artifacts(messages, session_id, existing_ids)

            result_text = self._render_message_content(final_message)

            return {
                "text": result_text,
                "plot": None,
                "artifacts": artifacts,
            }
        except (asyncio.TimeoutError, TimeoutError) as e:
            print(f"⏱️ Timeout error during agent analysis: {str(e)}")
            reset_agent()
            raise
        except Exception as e:
            print(f"❌ Unexpected error during agent analysis: {str(e)}")
            reset_agent()
            raise

    # ============================================================================
    # Artifact Helpers
    # ============================================================================

    async def _capture_and_cache_artifacts(
        self,
        messages: List[Any],
        session_id: str,
        existing_ids: set[str],
    ) -> List[Dict[str, Any]]:
        """Extract publish_artifact results, fetch their resources, and cache new artifacts locally."""
        if not messages:
            return []

        tool_calls_by_id: Dict[str, str] = {}
        for message in messages:
            if isinstance(message, AIMessage):
                for tool_call in message.tool_calls or []:
                    tool_calls_by_id[tool_call["id"]] = tool_call.get("name", "")

        artifact_descriptors: List[Dict[str, Any]] = []

        for message in messages:
            if not isinstance(message, ToolMessage):
                continue

            tool_name = tool_calls_by_id.get(message.tool_call_id, "")
            if not tool_name or not tool_name.endswith("publish_artifact"):
                continue

            payloads = self._extract_structured_payloads(message)
            if not payloads:
                continue

            server_id = self.tool_server_map.get(tool_name)
            client = self.client_by_server.get(server_id) if server_id else None

            if client is None and self.client_by_server:
                # Fallback: use the first client if mapping fails.
                server_id, client = next(iter(self.client_by_server.items()))

            if client is None or server_id is None:
                continue

            for payload in payloads:
                artifact_id, metadata = await self._materialize_artifact(
                    payload,
                    session_id,
                    server_id,
                    client,
                )
                if artifact_id is None or metadata is None:
                    continue
                if artifact_id in existing_ids:
                    continue
                mime_type = metadata.get("mime_type") or ""
                display_type = "image" if isinstance(mime_type, str) and mime_type.lower().startswith("image/") else "download"
                artifact_descriptors.append(
                    {
                        "filename": metadata.get("filename"),
                        "mime_type": metadata.get("mime_type"),
                        "proxy_path": metadata.get("proxy_path"),
                        "uri": metadata.get("uri"),
                        "display_type": display_type,
                    }
                )

        return artifact_descriptors

    def _extract_structured_payloads(self, message: ToolMessage) -> List[Dict[str, Any]]:
        """Pull structured payloads from a ToolMessage."""
        payloads: List[Dict[str, Any]] = []
        artifact = getattr(message, "artifact", None)

        if isinstance(artifact, dict):
            structured = artifact.get("structured_content")
            if structured:
                if isinstance(structured, list):
                    payloads.extend(item for item in structured if isinstance(item, dict))
                elif isinstance(structured, dict):
                    payloads.append(structured)

        if payloads:
            return payloads

        content = message.content
        content_items: List[str] = []
        if isinstance(content, str):
            content_items.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    content_items.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    content_items.append(item["text"])

        for item in content_items:
            try:
                data = json.loads(item)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                payloads.append(data)
            elif isinstance(data, list):
                payloads.extend(d for d in data if isinstance(d, dict))

        return payloads

    async def _materialize_artifact(
        self,
        payload: Dict[str, Any],
        session_id: str,
        server_id: str,
        client: MultiServerMCPClient,
    ) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Fetch artifact content via MCP, materialize it, and cache it locally.

        The MCP server may return artifacts as raw bytes or as a JSON payload
        containing a base64-encoded ``blob`` key. This helper normalizes both
        cases into filesystem artifacts so the FastAPI layer can serve them.
        """
        uri = payload.get("uri")
        if not uri:
            return None, None

        try:
            blobs = await client.get_resources(server_name=server_id, uris=[uri])
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Failed to fetch MCP resource {uri}: {exc}")
            return None, None

        if not blobs:
            return None, None

        blob = blobs[0]
        try:
            data = blob.as_bytes()
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Unable to read blob bytes for {uri}: {exc}")
            return None, None

        if data.strip().startswith(b"{") and b'"blob"' in data:
            try:
                payload_json = json.loads(data.decode("utf-8"))
                blob_value = payload_json.get("blob")
                if isinstance(blob_value, str):
                    data = base64.b64decode(blob_value)
                mime_candidate = payload_json.get("mimeType")
                if isinstance(mime_candidate, str):
                    payload["mime_type"] = mime_candidate
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️ Failed to decode JSON blob for {uri}: {exc}")

        original_name = payload.get("filename") or Path(payload.get("path", "artifact")).name
        if not original_name:
            original_name = "artifact"

        mime_type = payload.get("mime_type")
        if not mime_type:
            mimetype_attr = getattr(blob, "mimetype", None)
            mime_type = mimetype_attr or "application/octet-stream"

        session_registry = self.artifact_registry.setdefault(session_id, {})
        for existing_id, metadata in session_registry.items():
            if metadata.get("uri") == uri:
                return existing_id, metadata

        artifact_id = uuid.uuid4().hex
        artifact_dir = ARTIFACT_CACHE_ROOT / session_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(original_name).suffix
        stored_name = f"{artifact_id}{suffix}" if suffix else artifact_id
        stored_path = artifact_dir / stored_name
        stored_path.write_bytes(data)

        proxy_path = f"/artifacts/{session_id}/{artifact_id}"

        session_registry[artifact_id] = {
            "path": str(stored_path),
            "filename": original_name,
            "mime_type": mime_type,
            "proxy_path": proxy_path,
            "uri": uri,
        }

        return artifact_id, session_registry[artifact_id]

    def get_cached_artifact_metadata(self, session_id: str, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Expose cached artifact metadata for FastAPI handlers."""
        return self.artifact_registry.get(session_id, {}).get(artifact_id)

# ============================================================================
# Global Agent Instance
# ============================================================================

_agent_instance: Optional[DataAnalystAgent] = None

async def get_agent() -> DataAnalystAgent:
    """
    Gets or creates a global agent instance with lazy initialization.

    Returns:
        The global agent instance.
    """
    global _agent_instance
    if _agent_instance is None:
        runtime_cfg = _load_runtime_config()
        _agent_instance = DataAnalystAgent(
            runtime_cfg["mcp_endpoints"],
            runtime_cfg["llm"],
            runtime_cfg["retries"],
        )

    # Lazy initialize on first use
    if _agent_instance.agent is None and not _agent_instance._initializing:
        await _agent_instance.initialize()

    return _agent_instance


def reset_agent() -> None:
    """
    Resets the global agent instance.
    
    This function should be called when the agent encounters an error state
    (e.g., timeout) from which it cannot recover. It clears the global agent
    instance so that the next call to get_agent() will reinitialize it.
    """
    global _agent_instance
    if _agent_instance is not None:
        print("🔄 Clearing agent instance for reset...")
        _agent_instance = None
    else:
        print("ℹ️ Agent instance was already None")
